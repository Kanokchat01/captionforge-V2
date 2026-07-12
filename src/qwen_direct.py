"""
v6 primary caption engine — "qwen_direct".

One multimodal call per style: uniform frames go straight to the vision
model together with a short imperative persona, and the model answers with
the caption inside <caption_output> tags. No scene report, no Best-of-N,
no LLM judge anywhere on this path — that geometry is board-verified at
0.92-0.93 (see prompts.py). Layered on top are deterministic, code-only
guards the recipe itself lacks:

  tag missing/empty          -> one identical retry
  style_violations() regex   -> one regeneration carrying the violation list
  model down after retries   -> the identical call on the spare model
  everything failed          -> "" (main.py fills its never-empty fallback)
"""
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

import config
from fireworks_vision_client import extract_frames_b64
from prompts import (
    QWEN_DIRECT_SYSTEM_PROMPT,
    build_qwen_direct_prompt,
    extract_caption_tag,
    sanitize_caption,
    style_violations,
)

# Don't spend optional (retry/regen/spare) calls this close to the deadline —
# an imperfect caption already in hand beats a timeout-truncated run.
EXTRA_CALL_TIME_MARGIN_SECONDS = 25


def _style_temperature(style: str) -> float:
    override = config.QWEN_DIRECT_TEMPERATURE_BY_STYLE.get(style)
    return override if override is not None else config.QWEN_DIRECT_TEMPERATURE


def _messages(frames_b64: List[str], style: str, extra_note: str = "") -> list:
    """Frames first, persona text last — same part ordering as the verified
    recipe. `extra_note` carries the one-line violation feedback on a
    regeneration and must stay a single appended sentence (geometry rule)."""
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    ]
    text = build_qwen_direct_prompt(style)
    if extra_note:
        text += "\n" + extra_note
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": QWEN_DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _call(captioner, frames_b64: List[str], style: str, model: str,
          extra_note: str = "") -> str:
    """One styled vision call -> extracted caption ("" if the tag is absent).
    Transport-level retries (429/5xx/timeouts) live inside _chat_with_retry."""
    raw = captioner._chat_with_retry(
        messages=_messages(frames_b64, style, extra_note),
        model=model,
        max_tokens=config.QWEN_DIRECT_MAX_TOKENS,
        temperature=_style_temperature(style),
        timeout_seconds=config.QWEN_DIRECT_TIMEOUT_SECONDS,
    )
    caption = extract_caption_tag(raw)
    if not caption:
        print(f"[qwen-direct] {style}: no <caption_output> tag in reply: {raw[:160]!r}")
    return caption


def _caption_one_style(captioner, frames_b64: List[str], style: str,
                       time_remaining: Callable[[], float]) -> str:
    def clock_allows() -> bool:
        return time_remaining() > EXTRA_CALL_TIME_MARGIN_SECONDS

    caption = ""
    # Primary model, plus one identical content-level retry if the reply
    # carried no tag (transport failures already retried inside the call).
    try:
        caption = _call(captioner, frames_b64, style, config.QWEN_DIRECT_MODEL)
        if not caption and clock_allows():
            print(f"[qwen-direct] {style}: retrying once for a tagged caption")
            caption = _call(captioner, frames_b64, style, config.QWEN_DIRECT_MODEL)
    except Exception as e:
        print(f"[qwen-direct] {style}: primary model failed ({e})")

    # Spare tire: identical call on the spare model, only when the primary
    # produced nothing at all.
    if not caption and clock_allows():
        try:
            print(f"[qwen-direct] {style}: trying spare model {config.QWEN_DIRECT_SPARE_MODEL}")
            caption = _call(captioner, frames_b64, style, config.QWEN_DIRECT_SPARE_MODEL)
        except Exception as e:
            print(f"[qwen-direct] {style}: spare model failed ({e})")

    if not caption or config.QWEN_DIRECT_GUARD_LEVEL < 1:
        return caption

    # Guard level 1: deterministic cleanup + at most ONE violation-driven
    # regeneration. Keep the regenerated caption only if it actually fixes
    # the violations without introducing new ones.
    caption = sanitize_caption(caption)
    violations = style_violations(style, caption)
    if violations and clock_allows():
        note = ("Your previous caption had these problems: "
                f"{'; '.join(violations)} — fix exactly these.")
        print(f"[qwen-direct] {style}: regenerating once ({'; '.join(violations)})")
        try:
            regenerated = sanitize_caption(
                _call(captioner, frames_b64, style, config.QWEN_DIRECT_MODEL, extra_note=note))
            if regenerated and not style_violations(style, regenerated):
                caption = regenerated
        except Exception as e:
            print(f"[qwen-direct] {style}: regeneration failed ({e}) — keeping original")
    return caption


def caption_styles(captioner, frames_b64: List[str], styles: List[str],
                   time_remaining: Callable[[], float]) -> Dict[str, str]:
    """All requested styles for one already-extracted frame set, in parallel.
    A single style's failure returns "" for that style only. Also the unit
    main.py's salvage pass re-runs for styles that shipped a fallback."""
    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(styles))) as pool:
        futures = {pool.submit(_caption_one_style, captioner, frames_b64, s,
                               time_remaining): s for s in styles}
        for future, style in futures.items():
            try:
                results[style] = future.result()
            except Exception:
                print(f"[qwen-direct] {style}: unexpected failure:\n{traceback.format_exc()}")
                results[style] = ""
    return results


def caption_clip_qwen_direct(captioner, ensure_downloaded: Callable[[], str],
                             styles: List[str],
                             time_remaining: Callable[[], float],
                             frames_sink: "dict | None" = None) -> Dict[str, str]:
    """All requested styles for one clip, in parallel. A single style's
    failure returns "" for that style only; raises only when the frames
    stage itself fails (caller falls back for the whole clip). `frames_sink`,
    when given, receives the extracted frames under "frames_b64" so the
    caller's salvage pass can retry styles later without re-downloading."""
    local_path = ensure_downloaded()
    t0 = time.monotonic()
    frames_b64 = extract_frames_b64(local_path, config.QWEN_DIRECT_FRAMES,
                                    config.QWEN_DIRECT_FRAME_MAX_WIDTH)
    print(f"[qwen-direct] {len(frames_b64)} frames @{config.QWEN_DIRECT_FRAME_MAX_WIDTH}px "
          f"in {time.monotonic() - t0:.2f}s")
    if frames_sink is not None:
        frames_sink["frames_b64"] = frames_b64
    return caption_styles(captioner, frames_b64, styles, time_remaining)
