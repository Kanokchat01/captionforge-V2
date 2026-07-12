"""
Local proxy for the official Track 2 judge.

Primary judge (v6): Gemini 2.5 Flash VISION via OpenRouter — it re-watches
4 uniform frames @1024px from the actual clip (the same geometry the writer
sees) and scores each caption on the official two axes, accuracy and style
match. This removes the old echo chamber where captions were scored against
reference reports written by our own Stage-1 model. Requires
OPENROUTER_API_KEY in .env — LOCAL ONLY, never bake this key into the
Docker image (the image is publicly pullable).

Secondary judge (legacy): two cross-family TEXT judges (glm-5p1 +
deepseek-v4-pro on Fireworks) scoring against eval/reference_reports.json.
Kept as a second opinion; degrades gracefully when refs are missing.

NEVER add a Qwen model as a judge here: the v6 writer IS Qwen, and a model
family scoring its own outputs is systematically generous.

The absolute numbers are proxies — their job is RELATIVE comparison between
pipeline variants (Δ). Ship decisions belong to the official board.

Usage:
    python scripts/local_eval.py output/results_v6.json
    python scripts/local_eval.py output/results_v6.json --judges vision
    python scripts/local_eval.py output/results_v6.json --tag r1 --tasks eval/eval_tasks.json

Writes <results>.eval.json next to the input with per-caption details.
Cost note: one full run = ~60 vision calls ≈ $0.03-0.08 on gemini-2.5-flash
($0.30/M in, $2.50/M out; roughly 3-5K input tokens per call).
"""
import argparse
import base64
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

# Windows consoles default to cp1252, which can't print emojis that may
# appear in judged captions — never let a print kill the report.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

import config  # noqa: E402
from prompts import STYLE_DESCRIPTIONS  # noqa: E402

REFS_PATH = os.path.join(ROOT, "eval", "reference_reports.json")
FRAME_CACHE_DIR = os.path.join(ROOT, "eval", "frame_cache")
JUDGE_TEXT_MODELS = [
    "accounts/fireworks/models/glm-5p1",
    "accounts/fireworks/models/deepseek-v4-pro",
]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_VISION_MODEL = os.environ.get("EVAL_VISION_MODEL", "google/gemini-2.5-flash")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
JUDGE_FRAMES = 4
JUDGE_FRAME_WIDTH = 1024
MAX_WORKERS = 8
STYLE_ORDER = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

EVAL_TEXT_SYSTEM_PROMPT = (
    "You are the automated judge of a video-captioning contest. You are given "
    "the contest's official one-line definition of a caption style, a verified "
    "ground-truth scene report describing everything confirmed to be in the "
    "video, and one caption. Score the caption on the contest's two official "
    "axes, each 0.0-1.0 independently:\n"
    "- accuracy: every claim in the caption is supported by the scene report. "
    "Penalize heavily: claims that contradict the report, specific details "
    "(counts, colors, actions, objects) the report does not support, quoted "
    "on-screen text, or a named real-world city/country/landmark. Vague but "
    "consistent captions are accurate but a caption that correctly names "
    "several specific details from the report deserves a higher accuracy "
    "score than one generic enough to fit many different videos.\n"
    "- style_match: the caption genuinely lands the official style definition "
    "(tone, intent, and for humorous styles, whether it is actually funny "
    "about THIS scene rather than generic filler).\n"
    'Respond with ONLY a JSON object: {"accuracy": <0-1>, "style_match": <0-1>}'
)

EVAL_VISION_SYSTEM_PROMPT = (
    "You are the automated judge of a video-captioning contest. You are shown "
    "several still frames sampled uniformly from the actual video clip — "
    "treat the frames as ground truth — plus the contest's official one-line "
    "definition of a caption style and one caption. Score the caption on the "
    "contest's two official axes, each 0.0-1.0 independently:\n"
    "- accuracy: every claim in the caption is supported by what the frames "
    "visibly show. Penalize heavily: claims contradicting the frames, "
    "specific details (counts, colors, actions, objects) the frames do not "
    "support, quoted on-screen text, or a named real-world city/country/"
    "landmark. A caption that correctly names several specific visible "
    "details deserves a higher accuracy score than one generic enough to fit "
    "many different videos.\n"
    "- style_match: the caption genuinely lands the official style definition "
    "(tone, intent, and for humorous styles, whether it is actually funny "
    "about THIS scene rather than generic filler).\n"
    'Respond with ONLY a JSON object: {"accuracy": <0-1>, "style_match": <0-1>}'
)


def _parse_scores(raw: str) -> dict:
    data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    return {
        "accuracy": max(0.0, min(1.0, float(data["accuracy"]))),
        "style_match": max(0.0, min(1.0, float(data["style_match"]))),
    }


def get_frames_b64(task_id: str, video_url: str) -> list:
    """4 uniform frames @1024px for a clip, cached as JPEGs under
    eval/frame_cache/<task_id>/ so repeated eval runs cost nothing."""
    cache_dir = os.path.join(FRAME_CACHE_DIR, task_id)
    cached = sorted(glob.glob(os.path.join(cache_dir, "frame_*.jpg")))
    if len(cached) >= 1:
        frames = []
        for p in cached[:JUDGE_FRAMES]:
            with open(p, "rb") as f:
                frames.append(base64.b64encode(f.read()).decode("ascii"))
        return frames
    from downloader import download_video  # noqa: E402 — src path injected above
    from fireworks_vision_client import _extract_frames  # noqa: E402
    os.makedirs(cache_dir, exist_ok=True)
    local_path = download_video(video_url)
    # _extract_frames leaves frame_NNN.jpg files in cache_dir — that IS the cache.
    frames = _extract_frames(local_path, JUDGE_FRAMES, cache_dir, max_width=JUDGE_FRAME_WIDTH)
    return [base64.b64encode(jpeg).decode("ascii") for _, jpeg in frames]


def judge_vision_once(style: str, frames_b64: list, caption: str) -> dict:
    """One Gemini vision judgment via OpenRouter (frames are ground truth)."""
    content = [{
        "type": "text",
        "text": (
            f'Official style definition of "{style}": "{STYLE_DESCRIPTIONS[style]}"\n\n'
            f"Caption to score:\n{caption}\n\n"
            "The frames from the actual video follow."
        ),
    }]
    content += [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    ]
    last_exc = None
    for _ in range(2):
        try:
            resp = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={
                    "model": OPENROUTER_VISION_MODEL,
                    "messages": [
                        {"role": "system", "content": EVAL_VISION_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    "max_tokens": 200,
                    "temperature": 0.0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return _parse_scores(resp.json()["choices"][0]["message"]["content"])
        except Exception as e:  # noqa: BLE001 — retry once, then surface
            last_exc = e
    raise RuntimeError(f"vision judge failed twice: {last_exc}")


def judge_text_once(model: str, style: str, reference: str, caption: str) -> dict:
    user = (
        f'Official style definition of "{style}": "{STYLE_DESCRIPTIONS[style]}"\n\n'
        f"Verified ground-truth scene report:\n{reference}\n\n"
        f"Caption to score:\n{caption}"
    )
    last_exc = None
    for _ in range(2):
        try:
            resp = requests.post(
                f"{config.FIREWORKS_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {config.FIREWORKS_API_KEY}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": EVAL_TEXT_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 120,
                    "temperature": 0.0,
                    "reasoning_effort": "none",
                    "response_format": {"type": "json_object"},
                },
                timeout=45,
            )
            resp.raise_for_status()
            return _parse_scores(resp.json()["choices"][0]["message"]["content"])
        except Exception as e:  # noqa: BLE001 — retry once, then surface
            last_exc = e
    raise RuntimeError(f"judge {model} failed twice: {last_exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="path to a results.json produced by the pipeline")
    ap.add_argument("--tag", default="", help="label stored in the eval output")
    ap.add_argument("--tasks", default=os.path.join(ROOT, "eval", "eval_tasks.json"),
                    help="tasks.json with video_url per task_id (for the vision judge)")
    ap.add_argument("--judges", default="vision,text",
                    help="comma list: vision (Gemini/OpenRouter), text (Fireworks vs refs)")
    args = ap.parse_args()
    want = {j.strip() for j in args.judges.split(",") if j.strip()}

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    use_vision = "vision" in want
    if use_vision and not OPENROUTER_API_KEY:
        print("[eval] !!! OPENROUTER_API_KEY missing in .env — vision judge DISABLED, "
              "falling back to text judges only. Add the key for board-like scoring. !!!")
        use_vision = False

    refs = {}
    use_text = "text" in want
    if use_text:
        try:
            with open(REFS_PATH, encoding="utf-8") as f:
                refs = json.load(f)
        except OSError:
            print(f"[eval] no {REFS_PATH} — text judges disabled")
            use_text = False

    url_by_task = {}
    if use_vision:
        with open(args.tasks, encoding="utf-8") as f:
            url_by_task = {t["task_id"]: t["video_url"] for t in json.load(f)}

    if not use_vision and not use_text:
        print("[eval] no judge available (need OPENROUTER_API_KEY and/or reference reports)")
        return 1

    # Pre-fetch frames sequentially (downloads may be slow on home bandwidth;
    # cache makes every later run instant).
    frames_by_task = {}
    if use_vision:
        for row in results:
            tid = row.get("task_id")
            if tid not in url_by_task:
                print(f"[eval] no video_url for {tid} — vision judge skipped for it")
                continue
            try:
                frames_by_task[tid] = get_frames_b64(tid, url_by_task[tid])
            except Exception as e:  # noqa: BLE001
                print(f"[eval] frames for {tid} failed ({e}) — vision judge skipped for it")

    jobs = []  # (task_id, style, caption)
    for row in results:
        tid = row.get("task_id")
        for style, caption in row.get("captions", {}).items():
            if style in STYLE_DESCRIPTIONS and caption:
                jobs.append((tid, style, caption))
    if not jobs:
        print("[eval] nothing to score")
        return 1

    def score(job):
        tid, style, caption = job
        entry: dict[str, Any] = {"task_id": tid, "style": style, "caption": caption,
                                 "vision": None, "text": None}
        if use_vision and tid in frames_by_task:
            try:
                entry["vision"] = judge_vision_once(style, frames_by_task[tid], caption)
            except Exception as e:  # noqa: BLE001
                print(f"[eval] vision judge failed for {tid}/{style}: {e}")
        ref = refs.get(tid, {}).get("verified_report") if use_text else None
        if ref:
            try:
                per_judge = {m.rsplit("/", 1)[-1]: judge_text_once(m, style, ref, caption)
                             for m in JUDGE_TEXT_MODELS}
                entry["text"] = {
                    "accuracy": sum(j["accuracy"] for j in per_judge.values()) / len(per_judge),
                    "style_match": sum(j["style_match"] for j in per_judge.values()) / len(per_judge),
                    "judges": per_judge,
                }
            except Exception as e:  # noqa: BLE001
                print(f"[eval] text judges failed for {tid}/{style}: {e}")
        # Primary = vision when available (closest to how the board judges).
        primary = entry["vision"] or entry["text"]
        if primary:
            entry["accuracy"] = primary["accuracy"]
            entry["style_match"] = primary["style_match"]
            entry["score"] = (primary["accuracy"] + primary["style_match"]) / 2
        return entry

    n_judges = (1 if use_vision else 0) + (len(JUDGE_TEXT_MODELS) if use_text else 0)
    print(f"[eval] scoring {len(jobs)} captions (vision={use_vision}, text={use_text}, "
          f"{n_judges} judge model(s))...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        scored = [s for s in pool.map(score, jobs) if s.get("score") is not None]
    if not scored:
        print("[eval] every judgment failed")
        return 1

    # --- per-clip x per-style table (primary scores) ---
    by_task: dict = {}
    for s in scored:
        by_task.setdefault(s["task_id"], {})[s["style"]] = s
    styles = [st for st in STYLE_ORDER if any(st in v for v in by_task.values())]
    col = 18
    print("\n" + "clip".ljust(8) + "".join(st[:col - 2].ljust(col) for st in styles) + "mean")
    for tid in sorted(by_task):
        row_scores = []
        line = str(tid).ljust(8)
        for st in styles:
            s = by_task[tid].get(st)
            if s:
                line += f"a{s['accuracy']:.2f}/s{s['style_match']:.2f}".ljust(col)
                row_scores.append(s["score"])
            else:
                line += "-".ljust(col)
        line += f"{sum(row_scores) / len(row_scores):.3f}" if row_scores else "-"
        print(line)

    # --- aggregates, split by judge family ---
    def aggregate(key: str) -> "tuple[float, float, int] | None":
        rows = [s[key] for s in scored if s.get(key)]
        if not rows:
            return None
        acc = sum(r["accuracy"] for r in rows) / len(rows)
        sty = sum(r["style_match"] for r in rows) / len(rows)
        return acc, sty, len(rows)

    print("\nper-style means (primary judge):")
    for st in styles:
        ss = [s for s in scored if s["style"] == st]
        acc = sum(s["accuracy"] for s in ss) / len(ss)
        sty = sum(s["style_match"] for s in ss) / len(ss)
        print(f"  {st:<20} accuracy {acc:.3f}   style {sty:.3f}   combined {(acc + sty) / 2:.3f}")

    for key, label in (("vision", "VISION (Gemini, frames)"), ("text", "TEXT (Fireworks, refs)")):
        agg = aggregate(key)
        if agg:
            acc, sty, n = agg
            print(f"\n{label}: {(acc + sty) / 2:.4f}   (accuracy {acc:.4f}, style {sty:.4f}, {n} captions)")

    overall = sum(s["score"] for s in scored) / len(scored)
    print(f"\nOVERALL (primary): {overall:.4f}   ({len(scored)} captions)")

    print("\n5 weakest captions (primary):")
    for s in sorted(scored, key=lambda x: x["score"])[:5]:
        print(f"  [{s['score']:.2f} a{s['accuracy']:.2f}/s{s['style_match']:.2f}] "
              f"{s['task_id']}/{s['style']}: {s['caption'][:110]}")

    out_path = args.results.rsplit(".", 1)[0] + ".eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "overall": overall,
                   "vision_model": OPENROUTER_VISION_MODEL if use_vision else None,
                   "captions": scored}, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] details written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
