"""
Central config. All values overridable via environment variables so the
Track 2 rules ("no restriction, use your own credentials") are respected —
nothing is hardcoded, nothing bundled into the image.
"""
import os
import socket

# Auto-add winget-installed Gyan.FFmpeg to PATH on Windows to support local testing
if os.name == "nt":
    winget_packages_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.exists(winget_packages_dir):
        for root, dirs, files in os.walk(winget_packages_dir):
            if "ffmpeg.exe" in files:
                os.environ["PATH"] = root + os.pathsep + os.environ["PATH"]
                break

# Set socket timeout to prevent indefinite hangs in requests stream iteration
socket.setdefaulttimeout(40)

KEEP_DOWNLOADS = os.environ.get("KEEP_DOWNLOADS", "false").lower() == "true"

# --- Required for the primary (base caption) pass ---
# Model roles were chosen by a head-to-head benchmark (2026-07-11, 3 sample
# clips, cross-judged by glm-5p1 + deepseek-v4-pro on the official rubric):
#   Stage 1 eyes    -> minimax-m3     (ONLY account model that accepts whole
#                                      videos via video_url — verified
#                                      2026-07-11, ~4s/clip; sees motion,
#                                      timelines, camera movement that still
#                                      frames can't. NOT trusted for OCR: it
#                                      confidently misread a real building
#                                      sign, so prompts ban transcribing
#                                      on-screen text. No audio arrives.)
#   Stage 1 fallback-> kimi-k2p7-code      (frame-based; most detailed, meme-aware
#                                      scene reports; hallucination-free vs
#                                      real frames. kimi/qwen reject video_url
#                                      with "videos limited to 0".)
#   Stage 2 caption -> glm-5p2        (best caption writer: 0.874 vs 0.850 qwen,
#                                      0.830 kimi-k2p7-code, 0.666 minimax-m3 —
#                                      minimax stays out of the WRITER role;
#                                      it also failed JSON output on 2/3 clips)
#   Judge/polish    -> qwen3p7-plus   (runner-up, fastest, different family from
#                                      the writer to avoid self-preference bias)
ENABLE_NATIVE_VIDEO = os.environ.get("ENABLE_NATIVE_VIDEO", "true").lower() == "true"
FIREWORKS_NATIVE_VIDEO_MODEL = os.environ.get("FIREWORKS_NATIVE_VIDEO_MODEL", "accounts/fireworks/models/minimax-m3")
# Whole-clip analysis of a 2-min UHD video (Fireworks fetches the URL
# server-side) needs more headroom than a frame call.
FIREWORKS_NATIVE_VIDEO_TIMEOUT_SECONDS = float(os.environ.get("FIREWORKS_NATIVE_VIDEO_TIMEOUT_SECONDS", "90"))
# Stage 1.5: after the native scene report, re-attach the video and have the
# model verify its own report, deleting claims it can't confirm (PADAYON-style
# self-correction). Single attempt, sanity-gated, falls back to the unverified
# report on any failure — it can only remove hallucinations, never add them.
ENABLE_REPORT_VERIFICATION = os.environ.get("ENABLE_REPORT_VERIFICATION", "true").lower() == "true"
FIREWORKS_VERIFY_TIMEOUT_SECONDS = float(os.environ.get("FIREWORKS_VERIFY_TIMEOUT_SECONDS", "45"))
# Skip the verification pass when the global clock is this low — captions from
# an unverified report still beat fallback captions from a timed-out clip.
VERIFY_MIN_TIME_REMAINING_SECONDS = float(os.environ.get("VERIFY_MIN_TIME_REMAINING_SECONDS", "180"))
FIREWORKS_VISION_MODEL = os.environ.get("FIREWORKS_VISION_MODEL", "accounts/fireworks/models/kimi-k2p7-code")
# Used when the primary vision model fails on a clip (degrade chain).
FIREWORKS_VISION_FALLBACK_MODEL = os.environ.get("FIREWORKS_VISION_FALLBACK_MODEL", "accounts/fireworks/models/qwen3p7-plus")
FIREWORKS_TEXT_MODEL = os.environ.get("FIREWORKS_TEXT_MODEL", "accounts/fireworks/models/glm-5p2")
# Frame sampling is adaptive: one frame every SECONDS_PER_FRAME, clamped to
# [MIN_FRAMES_PER_CLIP, MAX_FRAMES_PER_CLIP]. Hidden eval clips are 30s-2min,
# so this yields 8 frames for short clips up to 15-16 for 2-minute ones.
MIN_FRAMES_PER_CLIP = int(os.environ.get("MIN_FRAMES_PER_CLIP", "8"))
MAX_FRAMES_PER_CLIP = int(os.environ.get("MAX_FRAMES_PER_CLIP", "16"))
SECONDS_PER_FRAME = float(os.environ.get("SECONDS_PER_FRAME", "8"))
# Downscale before base64-encoding to avoid write timeouts on large resolutions.
FIREWORKS_FRAME_MAX_WIDTH = int(os.environ.get("FIREWORKS_FRAME_MAX_WIDTH", "768"))
# Dedicated timeout for vision calls.
FIREWORKS_VISION_TIMEOUT_SECONDS = float(os.environ.get("FIREWORKS_VISION_TIMEOUT_SECONDS", "60"))

# --- Optional secondary pass: judge pick-best/polish/self-critique via
#     Fireworks. Must never block or crash the primary submission if
#     unavailable. ---
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_JUDGE_MODEL = os.environ.get("FIREWORKS_JUDGE_MODEL", "accounts/fireworks/models/qwen3p7-plus")

# Blanket polish is OFF by default: with Best-of-N + judge selection in
# place, unconditionally rewriting the winning caption risks making it worse
# or breaking the per-style word-count rules. The self-critique loop below
# still polishes any caption the judge scores under the threshold.
ENABLE_JUDGE_POLISH = os.environ.get("ENABLE_JUDGE_POLISH", "false").lower() == "true"
ENABLE_SELF_CRITIQUE = os.environ.get("ENABLE_SELF_CRITIQUE", "true").lower() == "true"
# Best-of-N candidate caption sets per clip in Stage 2 (judge picks per
# style). 5 since 2026-07-11: Track 2 has no token budget and the 3-set run
# finished 3 clips in 81s of the 540s budget, so wider sampling is free —
# more shots at a genuinely funny line for the humor styles.
BEST_OF_N = int(os.environ.get("BEST_OF_N", "5"))
MAX_CRITIQUE_RETRIES = int(os.environ.get("MAX_CRITIQUE_RETRIES", "2"))
CRITIQUE_PASS_THRESHOLD = float(os.environ.get("CRITIQUE_PASS_THRESHOLD", "8"))

# --- Orchestration / time-budget (hard rule: whole container <= 10 minutes) ---
INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json" if os.path.exists("/input/tasks.json") else "input/tasks.json")
OUTPUT_PATH = os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json" if os.path.exists("/output") else "output/results.json")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "6"))
# Hard limit is 600s (10 min). Leave a safety margin for the final JSON write.
TOTAL_BUDGET_SECONDS = float(os.environ.get("TOTAL_BUDGET_SECONDS", "540"))
# Per-request timeout for text (caption/judge) calls. NOT a contest rule —
# Track 2 only caps total runtime at 10 min, with no per-request limit; this
# is a self-chosen bound to keep any single stuck call from eating the budget.
# (Vision calls use the longer FIREWORKS_VISION_TIMEOUT_SECONDS above.)
PER_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("PER_REQUEST_TIMEOUT_SECONDS", "28"))

# Reserved purely for writing the final results.json + process exit. If a
# worker is still running when (deadline - this) is reached, its result is
# NOT waited for any further — main.py fills a fallback immediately instead.
FINALIZATION_RESERVE_SECONDS = float(os.environ.get("FINALIZATION_RESERVE_SECONDS", "30"))

# If a task is picked up by a worker with less than this much time left on
# the global clock, don't even start the (expensive) captioning call — go
# straight to a fallback caption. Prevents starting doomed work.
CRITICAL_TIME_THRESHOLD_SECONDS = float(os.environ.get("CRITICAL_TIME_THRESHOLD_SECONDS", "45"))

REQUIRED_STYLES = {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}

# --- v6 primary engine: qwen_direct (one multimodal call per style) ---
# CAPTION_ASSEMBLY selects the caption engine:
#   qwen_direct (default) — uniform frames go straight to the vision model,
#       ONE call per style, caption extracted from <caption_output> tags.
#       No describe stage, no Best-of-N, no judge, no critique.
#   legacy_v5 — the previous scene-report -> Best-of-N -> judge pipeline
#       (kept for rollback only).
CAPTION_ASSEMBLY = (os.environ.get("CAPTION_ASSEMBLY") or "qwen_direct").strip().lower()
if CAPTION_ASSEMBLY not in ("qwen_direct", "legacy_v5"):
    print(f"[config] unknown CAPTION_ASSEMBLY={CAPTION_ASSEMBLY!r} — using qwen_direct")
    CAPTION_ASSEMBLY = "qwen_direct"


def _env_int(name: str, default: int) -> int:
    # Docker ARG defaults arrive as empty strings — treat "" as unset.
    return int(os.environ.get(name) or default)


QWEN_DIRECT_MODEL = os.environ.get("QWEN_DIRECT_MODEL") or "accounts/fireworks/models/qwen3p7-plus"
# Spare tire: the identical call on a different model family, used only after
# the primary model has failed all transport retries for a style.
QWEN_DIRECT_SPARE_MODEL = os.environ.get("QWEN_DIRECT_SPARE_MODEL") or "accounts/fireworks/models/kimi-k2p7-code"
QWEN_DIRECT_FRAMES = _env_int("QWEN_DIRECT_FRAMES", 4)
QWEN_DIRECT_FRAME_MAX_WIDTH = _env_int("QWEN_DIRECT_FRAME_MAX_WIDTH", 1024)
QWEN_DIRECT_MAX_TOKENS = _env_int("QWEN_DIRECT_MAX_TOKENS", 400)
QWEN_DIRECT_TEMPERATURE = float(os.environ.get("QWEN_DIRECT_TEMPERATURE") or 0.7)
QWEN_DIRECT_TIMEOUT_SECONDS = float(os.environ.get("QWEN_DIRECT_TIMEOUT_SECONDS") or 45)
# 0 = pure recipe parity (tag extraction + one retry + never-empty only);
# 1 = adds sanitize_caption + one style_violations()-driven regeneration.
QWEN_DIRECT_GUARD_LEVEL = _env_int("QWEN_DIRECT_GUARD_LEVEL", 1)


def _qd_style_temp(style: str):
    v = os.environ.get(f"QWEN_DIRECT_TEMP_{style.upper()}", "")
    return float(v) if v else None


# Optional per-style temperature overrides (experiment R2: formal cold,
# humor hot). None = use QWEN_DIRECT_TEMPERATURE.
QWEN_DIRECT_TEMPERATURE_BY_STYLE = {s: _qd_style_temp(s) for s in sorted(REQUIRED_STYLES)}
