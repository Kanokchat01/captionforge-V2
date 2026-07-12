"""
Local web demo for CaptionForge.

NOT part of the Track 2 submission -- the real submission runs headless in
Docker (see ../src/main.py: reads /input/tasks.json, writes
/output/results.json, no UI). This Flask app exists to try the pipeline
interactively during development and to record the hackathon demo video.

It reuses the exact same pipeline code the submission uses
(fireworks_vision_client.FireworksCaptioner, judge_polish.JudgeAssistant,
downloader.py, config.py, prompts.py) -- including the same Best-of-N
selection, word-count guard, and self-critique loop -- so what you see here
matches what the judged container would produce for the same clip.

/api/generate streams newline-delimited JSON (NDJSON) so the UI can show real
pipeline progress instead of a spinner: the clip takes 20-90s to process.

Run locally:
    pip install -r web_demo/requirements.txt
    python web_demo/app.py
Then open http://localhost:5000

Needs the same .env as the main pipeline (FIREWORKS_API_KEY required).
"""
import json
import os
import sys
import time
import traceback
from typing import Iterator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

from flask import Flask, Response, request, jsonify, render_template

import config
from downloader import download_video
from fireworks_vision_client import FireworksCaptioner, SceneAnalysisFailed
from judge_polish import JudgeAssistant
from prompts import (
    STYLE_WORD_RANGES,
    has_tech_jargon,
    in_word_range,
    sanitize_caption,
    stable_variety_index,
    word_count,
)

app = Flask(__name__)

ALL_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
HUMOR_STYLES_FOR_POLISH = {"sarcastic", "humorous_tech", "humorous_non_tech"}

# The three official example clips from the Participant Guide, offered as
# one-click presets so the demo doesn't require hunting for a URL.
EXAMPLE_CLIPS = [
    {"label": "Urban autumn boulevard",
     "url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4"},
    {"label": "Kitten in a garden",
     "url": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4"},
    {"label": "Office worker at a desk",
     "url": "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4"},
]

_captioner = None
_judge = None


def get_clients():
    """Lazy singleton init so a missing API key surfaces as a clean error on
    the first request instead of killing the server at import time."""
    global _captioner, _judge
    if _captioner is None:
        _captioner = FireworksCaptioner()
    if _judge is None:
        _judge = JudgeAssistant()
    return _captioner, _judge


def ndjson(event: str, **payload) -> str:
    """One newline-delimited JSON event for the streaming response."""
    return json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n"


def caption_meta(style: str, text: str) -> dict:
    """Word-count compliance info shown per caption in the UI — the same
    check the pipeline's word-count guard applies internally."""
    lo, hi = STYLE_WORD_RANGES.get(style, (15, 25))
    return {
        "words": word_count(text),
        "min_words": lo,
        "max_words": hi,
        "in_range": in_word_range(style, text),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        examples=EXAMPLE_CLIPS,
        models={
            "vision": config.FIREWORKS_VISION_MODEL.rsplit("/", 1)[-1],
            "text": config.FIREWORKS_TEXT_MODEL.rsplit("/", 1)[-1],
            "judge": config.FIREWORKS_JUDGE_MODEL.rsplit("/", 1)[-1],
        },
        best_of_n=config.BEST_OF_N,
    )


@app.route("/api/health")
def health():
    """Lets the UI warn about a missing key before the user waits 90s."""
    return jsonify({
        "api_key_set": bool(config.FIREWORKS_API_KEY),
        "self_critique": config.ENABLE_SELF_CRITIQUE,
        "best_of_n": config.BEST_OF_N,
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True, silent=True) or {}
    video_url = (data.get("video_url") or "").strip()
    requested = data.get("styles") or ALL_STYLES
    styles = [s for s in ALL_STYLES if s in requested] or ALL_STYLES

    if not video_url:
        return jsonify({"error": "Please provide a video URL."}), 400
    if not config.FIREWORKS_API_KEY:
        return jsonify({"error": "FIREWORKS_API_KEY is not set. Add it to your .env file."}), 500

    def stream() -> Iterator[str]:
        local_path = None
        t0 = time.monotonic()
        try:
            captioner, judge = get_clients()

            # --- Stage 1: download (lazy, same as main.py) ---------------
            # The native-video path sends the URL straight to Fireworks, so
            # the UHD file is only downloaded if the frame fallback runs.
            yield ndjson("stage", id="download", status="active")

            def ensure_downloaded() -> str:
                nonlocal local_path
                if local_path is None:
                    local_path = download_video(video_url)
                return local_path

            if not (config.ENABLE_NATIVE_VIDEO and video_url):
                ensure_downloaded()
            if local_path:
                size_mb = os.path.getsize(local_path) / (1024 * 1024)
                yield ndjson("stage", id="download", status="done", detail=f"{size_mb:.1f} MB")
            else:
                yield ndjson("stage", id="download", status="done",
                             detail="skipped — URL sent straight to the native video model")

            # --- Stage 2: scene report (native video, frame fallback) -> Best-of-N
            yield ndjson("stage", id="analyze", status="active")
            # variety_index is a legacy no-op (per-clip opening/emoji rotation
            # was removed 2026-07-12); kept so the interface stays stable.
            candidates, scene = captioner.caption_clip(
                ensure_downloaded, styles, video_url=video_url,
                variety_index=stable_variety_index(video_url))
            yield ndjson("stage", id="analyze", status="done",
                         detail=f"{len(candidates)} candidate set(s)")
            yield ndjson("scene", report=scene)

            # --- Stage 3: judge picks best, then self-critiques ---------
            yield ndjson("stage", id="judge", status="active")

            def polish_guarded(style, prompt_caption, baseline):
                """Same guard as main.py: revert a polish that pushes an
                in-range caption out of its style's word-count range."""
                polished = sanitize_caption(judge.polish(style, scene, prompt_caption))
                if (polished != baseline and in_word_range(style, baseline)
                        and not in_word_range(style, polished)):
                    return baseline
                return polished

            for style in styles:
                options = [c.get(style, "") for c in candidates if c.get(style)]
                # Same tech-jargon guard as main.py: humorous_non_tech is
                # defined as having none, so drop candidates that carry it.
                if style == "humorous_non_tech":
                    options = [o for o in options if not has_tech_jargon(o)] or options
                # Prefer candidates that already satisfy the word-count rule.
                compliant = [o for o in options if in_word_range(style, o)]
                pool = compliant or options

                if judge.available:
                    caption = judge.pick_best(style, scene, pool)
                else:
                    caption = next(iter(pool), "")
                result = caption

                if (config.ENABLE_JUDGE_POLISH and judge.available
                        and style in HUMOR_STYLES_FOR_POLISH):
                    result = polish_guarded(style, result, result)

                critique_score = None
                if config.ENABLE_SELF_CRITIQUE and judge.available:
                    for _ in range(config.MAX_CRITIQUE_RETRIES):
                        critique_score, feedback = judge.judge(style, scene, result)
                        if critique_score >= config.CRITIQUE_PASS_THRESHOLD:
                            break
                        hint = (f"{result} (reviewer feedback to address: {feedback})"
                                if feedback else result)
                        result = polish_guarded(style, hint, result)

                final = result or caption
                yield ndjson("caption", style=style, text=final,
                             candidates=len(pool), score=critique_score,
                             **caption_meta(style, final))

            yield ndjson("stage", id="judge", status="done")
            yield ndjson("done", elapsed_seconds=round(time.monotonic() - t0, 1))

        except SceneAnalysisFailed as e:
            yield ndjson("error", message=f"The vision model could not analyze this clip: {e}")
        except Exception as e:
            traceback.print_exc()
            yield ndjson("error", message=str(e))
        finally:
            # Match main.py: honour KEEP_DOWNLOADS so cached clips survive
            # between runs (re-downloading a UHD clip every request is slow).
            if local_path and os.path.exists(local_path) and not config.KEEP_DOWNLOADS:
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    # No stream_with_context needed: everything the generator uses (video_url,
    # styles) is read off the request above, so it never touches the request
    # context once streaming starts.
    return Response(stream(), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    # Local dev only. In a container the entrypoint is gunicorn (Dockerfile.web),
    # which imports `app` directly and never runs this block.
    if not config.FIREWORKS_API_KEY:
        print("[!] FIREWORKS_API_KEY is not set — the UI will load but generation will fail.")
    port = int(os.environ.get("PORT", "5000"))
    # 0.0.0.0 so the app is reachable when run inside a container.
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"[*] CaptionForge demo running at http://localhost:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
