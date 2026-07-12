"""
Builds VERIFIED ground-truth scene reports for the 15 official bucket clips
(eval/eval_tasks.json). These are what scripts/local_eval.py judges caption
accuracy against.

Per clip: Stage-1 native video report (minimax-m3) -> Stage-1.5 self-
verification pass (same model re-watches the clip and deletes unconfirmable
claims). Results cached in eval/reference_reports.json — a clip is only
rebuilt if its entry is missing or empty, so re-running is cheap. Delete an
entry (or the whole file) to force a rebuild.

Usage:  python scripts/build_reference.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from fireworks_vision_client import FireworksCaptioner  # noqa: E402

TASKS_PATH = os.path.join(ROOT, "eval", "eval_tasks.json")
CACHE_PATH = os.path.join(ROOT, "eval", "reference_reports.json")
MAX_WORKERS = 4  # gentle on rate limits; 15 clips x 2 calls finishes in a few minutes


def main() -> int:
    with open(TASKS_PATH, encoding="utf-8") as f:
        tasks = json.load(f)

    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)

    captioner = FireworksCaptioner()
    todo = [t for t in tasks if not cache.get(t["task_id"], {}).get("verified_report")]
    print(f"[refs] {len(tasks)} clips total, {len(tasks) - len(todo)} cached, building {len(todo)}")

    def build(task: dict):
        tid, url = task["task_id"], task["video_url"]
        name = os.path.basename(url)
        t0 = time.monotonic()
        report = captioner._scene_report_native(url, name)
        verified = captioner._verify_scene_report_native(url, report, name)
        print(f"[refs] {tid} ({name}) done in {time.monotonic() - t0:.1f}s "
              f"(report {len(report)} chars -> verified {len(verified)} chars)")
        return tid, {"video_url": url, "report": report, "verified_report": verified}

    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(build, t): t["task_id"] for t in todo}
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                tid, entry = fut.result()
                cache[tid] = entry
            except Exception as e:
                failures.append(tid)
                print(f"[refs] {tid} FAILED: {e}")

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"[refs] wrote {len(cache)} reference report(s) to {CACHE_PATH}")
    if failures:
        print(f"[refs] FAILED clips (re-run to retry): {', '.join(sorted(failures))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
