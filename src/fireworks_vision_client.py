"""
Primary captioner: Fireworks-hosted models.

Two-stage pipeline (no audio on any path — Fireworks' Whisper endpoints were
confirmed discontinued as of 2026-06-10, and video_url delivers no audio):
  Stage 1 (primary): the WHOLE clip goes to minimax-m3 via video_url —
           Fireworks fetches the URL server-side, so no local download is
           needed at all. Sees real motion/timelines/camera movement that
           still frames can't. OCR is untrusted (prompt bans transcribing
           on-screen text after it confidently misread a real sign).
  Stage 1 (degrade chain): adaptive frame sampling -> vision model
           (kimi-k2p7-code) full frames -> 4 frames -> 4 frames on
           qwen3p7-plus, so one flaky model/call never costs the clip.
  Stage 2: Best-of-N — the text model (glm-5p2, benchmark winner for
           caption writing) generates N candidate caption sets in parallel
           at different temperatures; main.py has the judge model pick the
           best per style.

caption_clip(video_source, styles, video_url=None)
    -> (candidates: list[dict], scene_report)
where candidates is a non-empty list of {style: caption} dicts and
video_source is a local path OR a zero-arg callable returning one (called
only if the frame fallback is actually needed — lazy download).
"""
import base64
import json
import re
import subprocess
import tempfile
import time
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union

import requests

import config
from prompts import (
    build_caption_generation_prompt,
    build_frame_scene_analysis_prompt,
    build_native_video_scene_analysis_prompt,
    build_report_verification_prompt,
    sanitize_caption,
)

MAX_API_RETRIES = 2
RETRY_BACKOFF_SECONDS = [3, 6]
RETRYABLE_MARKERS = ("503", "429", "unavailable", "rate limit", "resource_exhausted", "timeout", "deadline", "502", "504")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in RETRYABLE_MARKERS)


class SceneAnalysisFailed(Exception):
    """Raised when the model explicitly reports it could not analyze the
    frames (corrupted/blank/unreadable) rather than silently hallucinating."""
    pass


def _probe_duration_seconds(video_path: str) -> float:
    t0 = time.monotonic()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=30,
    )
    elapsed = time.monotonic() - t0
    try:
        duration = max(float(out.stdout.strip()), 0.1)
        print(f"[ffmpeg] Probed video duration: {duration:.2f}s (took {elapsed:.2f}s)")
        return duration
    except (ValueError, AttributeError) as e:
        print(f"[ffmpeg] Probed duration failed (took {elapsed:.2f}s): {e} (using 10s fallback)")
        return 10.0  # unknown duration fallback — still try to grab a few frames


def _extract_frames(video_path: str, max_frames: int, workdir: str, max_width: int = 0):
    """Seeks to `max_frames` evenly-spaced timestamps and grabs one JPEG
    frame at each via ffmpeg. Returns a list of (timestamp_seconds, jpeg_bytes).
    Uses per-timestamp -ss seeking (accurate, one process per frame) rather
    than a single fps= filter pass — max_frames is small (default 8) so the
    extra process overhead is negligible and timestamps come out exact.

    Frames are downscaled to `max_width` (falls back to
    config.FIREWORKS_FRAME_MAX_WIDTH, aspect-preserved, never upscaled)
    before saving — a real test against actual 1440p/4K source clips hit
    "write operation timed out" uploading un-resized native-resolution
    frames; multiple multi-MB JPEGs add up fast on ordinary home upload
    bandwidth. Downscaling first fixes that at the source instead of just
    raising timeouts."""
    duration = _probe_duration_seconds(video_path)
    if max_frames <= 1:
        timestamps = [duration / 2.0]
    else:
        # Avoid sampling frame 0 (often a black/blank first frame) and the
        # very last instant (can fail to decode on some containers).
        margin = duration * 0.03
        span_start, span_end = margin, max(duration - margin, margin + 0.1)
        step = (span_end - span_start) / (max_frames - 1) if max_frames > 1 else 0
        timestamps = [span_start + step * i for i in range(max_frames)]

    max_w = max_width or config.FIREWORKS_FRAME_MAX_WIDTH
    # scale filter: shrink to max_w wide if the source is wider, otherwise
    # leave as-is (never upscale a smaller source); height auto (-2 keeps it
    # divisible by 2, which some encoders require).
    scale_filter = f"scale='min({max_w},iw)':-2"

    print(f"[ffmpeg] Extracting up to {max_frames} frames from {os.path.basename(video_path)}...")
    t0 = time.monotonic()
    frames = []
    for i, ts in enumerate(timestamps):
        out_path = os.path.join(workdir, f"frame_{i:03d}.jpg")
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", video_path,
             "-frames:v", "1", "-vf", scale_filter, "-pix_fmt", "yuvj420p", "-strict", "-2", "-q:v", "4", out_path],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                frames.append((ts, f.read()))
        else:
            print(f"[ffmpeg] Warning: failed to extract frame {i} at timestamp {ts:.2f}s: {result.stderr.decode(errors='replace')}")

    elapsed = time.monotonic() - t0
    print(f"[ffmpeg] Extracted {len(frames)} frames successfully in {elapsed:.2f}s")
    if not frames:
        raise RuntimeError("ffmpeg failed to extract any frames from this clip")
    return frames


def extract_frames_b64(video_path: str, n_frames: int, max_width: int) -> List[str]:
    """Exactly `n_frames` uniform frames as base64 JPEG strings, in
    chronological order. Shared by the qwen_direct engine and
    scripts/local_eval.py so the local judge sees the same geometry the
    writer sees."""
    with tempfile.TemporaryDirectory() as workdir:
        frames = _extract_frames(video_path, n_frames, workdir, max_width=max_width)
    return [base64.b64encode(jpeg_bytes).decode("ascii") for _, jpeg_bytes in frames]


def _extract_json(text: str) -> dict:
    """Robust JSON extraction. Handles plain JSON, JSON with trailing junk
    ("Extra data" — some models emit two objects or commentary after the
    JSON even in json_object mode), a top-level JSON array (falls through to
    grab the first embedded object), and JSON embedded in surrounding prose."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        # Parsed fine but isn't an object (e.g. a top-level array/string) —
        # fall through to the object-extraction branches below.
    except json.JSONDecodeError:
        pass
    # Find the first '{' and raw_decode from there — tolerates trailing junk.
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from Fireworks response: {text[:300]}")


class FireworksCaptioner:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 vision_model: Optional[str] = None, text_model: Optional[str] = None):
        self.api_key = api_key or config.FIREWORKS_API_KEY
        if not self.api_key:
            raise ValueError("Missing FIREWORKS_API_KEY (required for the Fireworks captioner)")
        self.base_url = base_url or config.FIREWORKS_BASE_URL
        self.vision_model = vision_model or config.FIREWORKS_VISION_MODEL
        self.text_model = text_model or config.FIREWORKS_TEXT_MODEL

    def _chat_with_retry(self, messages: List[Dict[str, Any]], model: str, max_tokens: int = 1500,
                          response_format: Optional[Dict[str, Any]] = None, timeout_seconds: Optional[float] = None,
                          temperature: float = 0.5, attempts: Optional[int] = None):
        """`attempts` caps the TOTAL number of tries (1 = no retries at all);
        defaults to MAX_API_RETRIES + 1."""
        total_attempts = attempts if attempts is not None else MAX_API_RETRIES + 1
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # All models in our benchmark accept this and it keeps chain-of-
            # thought out of `content` (kimi models otherwise leak reasoning
            # text into the answer) while cutting latency roughly in half.
            "reasoning_effort": "none",
        }
        if response_format:
            payload["response_format"] = response_format
        timeout = timeout_seconds if timeout_seconds is not None else config.PER_REQUEST_TIMEOUT_SECONDS

        last_exc: Optional[Exception] = None
        for attempt in range(total_attempts):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers, json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_exc = e
                if attempt < total_attempts - 1 and _is_retryable(e):
                    delay = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
                    print(f"[retry] transient Fireworks vision error (attempt {attempt + 1}/{total_attempts}, "
                          f"waiting {delay}s): {e}")
                    time.sleep(delay)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def _scene_report(self, frames, duration: float, filename: str, model: str) -> str:
        timestamps = [ts for ts, _ in frames]
        prompt_text = build_frame_scene_analysis_prompt(timestamps, duration)

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for _, jpeg_bytes in frames:
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        print(f"[fireworks] Sending Scene Analysis request for {filename} to {model}...")
        t0 = time.monotonic()
        scene_report = self._chat_with_retry(
            messages=[{"role": "user", "content": content}],
            model=model,
            max_tokens=1200,
            timeout_seconds=config.FIREWORKS_VISION_TIMEOUT_SECONDS,
        )
        print(f"[fireworks] Scene Analysis completed in {time.monotonic() - t0:.2f}s")
        if scene_report.upper().startswith("ANALYSIS FAILED"):
            raise SceneAnalysisFailed(scene_report)
        return scene_report

    def _scene_report_native(self, video_url: str, filename: str) -> str:
        """Stage 1 primary path: the whole clip via video_url on the
        native-video model (minimax-m3). Fireworks fetches the URL
        server-side — no local download involved. max_tokens is generous
        because a real test at 800 got the report truncated mid-sentence."""
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": build_native_video_scene_analysis_prompt()},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]
        print(f"[fireworks] Sending NATIVE VIDEO Scene Analysis for {filename} "
              f"to {config.FIREWORKS_NATIVE_VIDEO_MODEL}...")
        t0 = time.monotonic()
        # attempts=2 (not the default 3): the worst case for a 90s-timeout
        # call is pathological, and the saved headroom funds the verification
        # pass; the frame degrade chain still backstops a double failure.
        # temperature 0.15: run-to-run report variance directly costs
        # accuracy — the judge re-watches the clip, and a low-temp report
        # stays closer to the modal viewing every other watcher gets.
        scene_report = self._chat_with_retry(
            messages=[{"role": "user", "content": content}],
            model=config.FIREWORKS_NATIVE_VIDEO_MODEL,
            max_tokens=2200,
            timeout_seconds=config.FIREWORKS_NATIVE_VIDEO_TIMEOUT_SECONDS,
            temperature=0.15,
            attempts=2,
        )
        print(f"[fireworks] Native video Scene Analysis completed in {time.monotonic() - t0:.2f}s")
        if scene_report.upper().startswith("ANALYSIS FAILED"):
            raise SceneAnalysisFailed(scene_report)
        return scene_report

    def _verify_scene_report_native(self, video_url: str, report: str, filename: str) -> str:
        """Stage 1.5 (PADAYON-style self-verification): re-watch the clip with
        the draft report attached and delete/soften/fix unconfirmable claims.
        Single attempt, no retries — on any failure, or a rewrite that no
        longer looks like the same report, the original is kept unchanged, so
        this pass can remove hallucinations but never lose the clip."""
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": build_report_verification_prompt(report)},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]
        print(f"[fireworks] Verifying scene report for {filename} against the video...")
        t0 = time.monotonic()
        try:
            verified = self._chat_with_retry(
                messages=[{"role": "user", "content": content}],
                model=config.FIREWORKS_NATIVE_VIDEO_MODEL,
                max_tokens=2400,
                timeout_seconds=config.FIREWORKS_VERIFY_TIMEOUT_SECONDS,
                temperature=0.2,
                attempts=1,
            )
        except Exception as e:
            print(f"[fireworks] report verification failed ({e}) — keeping unverified report")
            return report
        # Sanity gates: the rewrite must still be the same 10-section report,
        # not a summary, an apology, or a fragment.
        markers = ("SCENE REPORT", "SUBJECT", "RISKS")
        if all(m in verified for m in markers) and len(verified) >= 0.6 * len(report):
            print(f"[fireworks] report verification OK in {time.monotonic() - t0:.2f}s "
                  f"({len(report)} -> {len(verified)} chars)")
            return verified
        print("[fireworks] verification output failed sanity gates — keeping unverified report")
        return report

    def _generate_candidates(self, scene_report: str, styles: list, filename: str, n: int,
                             variety_index: int = 0) -> list:
        """Stage 2 Best-of-N: n candidate caption sets in parallel at
        different temperatures. Returns every set that parsed successfully
        (at least one, else raises the last error). `variety_index` is kept
        for interface compatibility only — see prompts.py."""
        prompt2 = build_caption_generation_prompt(scene_report, styles, variety_index)
        temperatures = [0.55, 0.7, 0.85, 1.0, 1.15][:max(1, n)] or [0.7]

        def one(temp: float):
            # max_tokens 1200: four fact-dense captions (formal up to 50
            # words) plus JSON overhead no longer fit in the old 600 — a
            # truncated response fails json parsing and silently costs a
            # whole candidate set.
            raw = self._chat_with_retry(
                messages=[{"role": "user", "content": prompt2}],
                model=self.text_model,
                max_tokens=1200,
                response_format={"type": "json_object"},
                temperature=temp,
            )
            captions = _extract_json(raw)
            # Key normalization (UniKL trick): models occasionally emit
            # "Humorous-Tech" or "humorous tech" — without this, that style
            # silently comes back empty for the whole candidate set.
            normalized = {
                str(k).strip().lower().replace("-", "_").replace(" ", "_"): v
                for k, v in captions.items()
            }
            # sanitize_caption also scrubs any prompt-instruction text the model
            # echoed into the caption (see prompts.sanitize_caption).
            return {s: sanitize_caption(str(normalized.get(s, ""))) for s in styles}

        print(f"[fireworks] Generating {len(temperatures)} caption candidate set(s) for {filename} via {self.text_model}...")
        t0 = time.monotonic()
        candidates, last_exc = [], None
        with ThreadPoolExecutor(max_workers=len(temperatures)) as pool:
            futures = [pool.submit(one, t) for t in temperatures]
            for f in futures:
                try:
                    cand = f.result()
                    if any(cand.get(s) for s in styles):
                        candidates.append(cand)
                except Exception as e:
                    last_exc = e
                    print(f"[fireworks] one caption candidate failed (tolerated): {e}")
        print(f"[fireworks] Caption Generation completed in {time.monotonic() - t0:.2f}s "
              f"({len(candidates)}/{len(temperatures)} candidates OK)")
        if not candidates:
            raise last_exc or RuntimeError("all caption candidates failed")
        return candidates

    def caption_clip(self, video_source: Union[str, Callable[[], str]], styles: list,
                     video_url: Optional[str] = None, variety_index: int = 0,
                     time_remaining: Optional[Callable[[], float]] = None):
        """Returns (candidates: list[dict], scene_report: str). Candidates
        is a non-empty list of {style: caption} dicts — the caller picks the
        best one per style (or just uses candidates[0]).

        `video_source` is a local file path OR a zero-arg callable returning
        one. The callable is invoked only if the frame fallback is actually
        needed, so the native-video path never waits on (or pays for) a
        local download at all. `time_remaining` (optional) reports seconds
        left on the caller's global clock; when given, the verification pass
        is skipped once the clock runs low.

        Stage 1 degrade chain so a flaky model/call never costs the clip:
        whole clip via video_url on the native-video model -> full frames on
        the primary vision model -> 4 frames on the primary -> 4 frames on
        the fallback vision model. A SceneAnalysisFailed from the NATIVE
        path also degrades to frames (Fireworks' server-side fetch can choke
        on a stream that local ffmpeg decodes fine); once the FRAME attempts
        report unreadable pixels we raise, as before."""
        filename = os.path.basename(video_url or (video_source if isinstance(video_source, str) else "clip"))
        scene_report = None

        if config.ENABLE_NATIVE_VIDEO and video_url:
            try:
                scene_report = self._scene_report_native(video_url, filename)
                # Stage 1.5: the model re-watches the clip and strips claims
                # it can't confirm. Only on the native path, only with clock
                # to spare — an unverified report still beats no captions.
                if (config.ENABLE_REPORT_VERIFICATION
                        and (time_remaining is None
                             or time_remaining() > config.VERIFY_MIN_TIME_REMAINING_SECONDS)):
                    scene_report = self._verify_scene_report_native(video_url, scene_report, filename)
            except Exception as e:
                print(f"[fireworks] Native video Scene Analysis failed ({e}) — degrading to frame analysis")

        if scene_report is None:
            local_path = video_source if isinstance(video_source, str) else video_source()
            filename = os.path.basename(local_path)
            with tempfile.TemporaryDirectory() as workdir:
                duration = _probe_duration_seconds(local_path)
                # Adaptive sampling: 1 frame per SECONDS_PER_FRAME, clamped.
                n_frames = max(config.MIN_FRAMES_PER_CLIP,
                               min(config.MAX_FRAMES_PER_CLIP, round(duration / config.SECONDS_PER_FRAME)))
                frames = _extract_frames(local_path, n_frames, workdir)

                attempts = [
                    (frames, self.vision_model),
                    (frames[:: max(1, len(frames) // 4)][:4], self.vision_model),
                    (frames[:: max(1, len(frames) // 4)][:4], config.FIREWORKS_VISION_FALLBACK_MODEL),
                ]
                last_exc = None
                for i, (attempt_frames, model) in enumerate(attempts):
                    try:
                        scene_report = self._scene_report(attempt_frames, duration, filename, model)
                        break
                    except SceneAnalysisFailed:
                        raise  # model explicitly says frames are unreadable — don't burn time retrying
                    except Exception as e:
                        last_exc = e
                        if i < len(attempts) - 1:
                            print(f"[fireworks] Scene Analysis attempt {i + 1} failed ({e}) — degrading")
                if scene_report is None:
                    raise last_exc or RuntimeError("scene analysis failed")

        candidates = self._generate_candidates(scene_report, styles, filename, config.BEST_OF_N,
                                               variety_index)
        return candidates, scene_report
