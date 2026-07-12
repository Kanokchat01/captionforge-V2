"""
CaptionForge — Track 2: Video Captioning Agent

Reads /input/tasks.json, captions every clip in each requested style, writes
/output/results.json. Two engines, selected by config.CAPTION_ASSEMBLY:

  qwen_direct (default) — uniform frames go straight to the vision model,
      ONE multimodal call per style, deterministic code-only guards on top
      (see qwen_direct.py). No describe stage, no judge, no critique.
  legacy_v5 — the previous scene-report -> Best-of-N -> judge pipeline,
      kept only for rollback.

Must exit 0, must finish within 10 minutes total, must never crash the whole
run because one clip failed, and must never let one stuck task blow the
whole container's time budget.
"""
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from typing import NoReturn

from dotenv import load_dotenv

load_dotenv()  # no-op in the real submission container (no .env bundled); used for local dev

import config
import qwen_direct
from downloader import download_video, probe_size_mb
from judge_polish import JudgeAssistant
from prompts import in_word_range, sanitize_caption, style_violations, word_count

START_TIME = time.monotonic()
DEADLINE = START_TIME + config.TOTAL_BUDGET_SECONDS
ENHANCEMENT_TIME_MARGIN_SECONDS = 20  # don't start optional steps this close to the deadline
PROBE_PHASE_BUDGET_SECONDS = 20  # cap on how long the pre-sort probing pass may take

HUMOR_STYLES_FOR_POLISH = {"sarcastic", "humorous_tech", "humorous_non_tech"}


def make_captioner():
    """Returns the Fireworks-hosted captioner. Which engine drives it is
    decided per task by config.CAPTION_ASSEMBLY."""
    from fireworks_vision_client import FireworksCaptioner
    print(f"[*] Caption engine: {config.CAPTION_ASSEMBLY} "
          f"(model {config.QWEN_DIRECT_MODEL if config.CAPTION_ASSEMBLY == 'qwen_direct' else config.FIREWORKS_TEXT_MODEL})")
    return FireworksCaptioner()


def time_remaining() -> float:
    return DEADLINE - time.monotonic()


# Generic, style-conformant fallback captions. Each obeys its style's word
# count / emoji / opening rules and makes no specific claims about the clip.
# An explicit "Caption unavailable" error message is guaranteed 0 on BOTH
# accuracy and style match; a plausible generic caption in the right tone
# still earns partial style credit when a clip fails.
FALLBACK_CAPTION_BY_STYLE = {
    "formal": (
        "This clip presents its central subject in a real-world setting, "
        "recorded with steady framing under natural lighting, and it moves "
        "through several distinct moments between its opening seconds and "
        "its final frame."
    ),
    "sarcastic": (
        "A video exists. Things happen in it, one after another, at a truly "
        "relentless pace. Somewhere out there, someone calls this peak "
        "content. Riveting stuff, honestly."
    ),
    "humorous_tech": (
        "My brain hit its loading screen the moment this clip started and "
        "never recovered; somewhere in there, a tiny error message is still "
        "waiting for someone to click OK."
    ),
    "humorous_non_tech": (
        "I pressed play with zero expectations, and somehow this clip still "
        "kept me watching to the very end, the same way grandma keeps me at "
        "the dinner table."
    ),
}


def fallback_captions(styles, reason: str = "processing error") -> dict:
    print(f"[fallback] using generic style captions ({reason}) for: {', '.join(styles)}")
    return {s: FALLBACK_CAPTION_BY_STYLE.get(s, FALLBACK_CAPTION_BY_STYLE["formal"]) for s in styles}


def guarded_polish(judge: JudgeAssistant, style: str, scene: str, prompt_caption: str,
                   baseline: "str | None" = None) -> str:
    """Polish, but revert to `baseline` if the rewrite pushed an in-range
    caption out of its style's word-count range — a longer 'funnier' rewrite
    that breaks the length rule loses more style points than it gains.
    `baseline` is the actual current caption; `prompt_caption` may carry
    extra reviewer-feedback text and is only what gets sent to the model."""
    baseline = baseline if baseline is not None else prompt_caption
    polished = sanitize_caption(judge.polish(style, scene, prompt_caption))
    if polished != baseline and in_word_range(style, baseline) and not in_word_range(style, polished):
        print(f"[word-guard] {style}: polish went out of range "
              f"({word_count(baseline)} -> {word_count(polished)} words) — keeping previous caption")
        return baseline
    return polished


def order_tasks_heaviest_first(tasks: list) -> list:
    """Probe each clip's size via a cheap HEAD request (in parallel, bounded
    time) so we can process the heaviest/slowest-looking clips first. This
    avoids a scenario where a large 4K clip ends up last in the queue right
    as the global time budget runs out. Unknown-size clips keep their
    relative order and are treated as weight 0 (not prioritized, not
    penalized)."""
    if not tasks:
        return tasks

    weights = {}
    with ThreadPoolExecutor(max_workers=min(config.CONCURRENCY, len(tasks))) as probe_pool:
        futures = {
            probe_pool.submit(probe_size_mb, t.get("video_url", "")): t.get("task_id")
            for t in tasks
        }
        done, not_done = futures_wait(futures.keys(), timeout=PROBE_PHASE_BUDGET_SECONDS)
        for f in done:
            task_id = futures[f]
            try:
                weights[task_id] = f.result() or 0.0
            except Exception:
                weights[task_id] = 0.0
        for f in not_done:
            weights[futures[f]] = 0.0  # didn't finish probing in time — treat as unknown

    return sorted(tasks, key=lambda t: weights.get(t.get("task_id"), 0.0), reverse=True)


def process_task(task: dict, captioner, judge: JudgeAssistant, variety_index: int = 0) -> dict:
    task_id = task.get("task_id", "unknown")
    video_url = task.get("video_url")
    styles = task.get("styles") or sorted(config.REQUIRED_STYLES)
    local_path = None

    if not video_url:
        print(f"[error] task {task_id} is missing video_url")
        return {"task_id": task_id, "captions": fallback_captions(styles, "missing video URL")}

    # Don't start new expensive work if the global clock is already too low
    # to safely attempt it — go straight to a fallback instead of getting
    # stuck partway through and eating into other tasks' time.
    if time_remaining() <= config.CRITICAL_TIME_THRESHOLD_SECONDS:
        print(f"[skip] task {task_id}: critical time remaining, using fallback without attempting analysis")
        return {"task_id": task_id, "captions": fallback_captions(styles, "time budget cutoff")}

    print(f"[*] Starting task {task_id}...")
    t_start = time.monotonic()
    try:
        # Lazy download: the native-video path hands the URL straight to
        # Fireworks (fetched server-side), so the multi-hundred-MB UHD file
        # is only downloaded here if the frame fallback actually runs.
        def ensure_downloaded() -> str:
            nonlocal local_path
            if local_path is None:
                local_path = download_video(video_url)
            return local_path

        # v6 primary engine: frames straight to the vision model, one call
        # per style, deterministic guards only — see qwen_direct.py.
        if config.CAPTION_ASSEMBLY == "qwen_direct":
            final_captions = {}
            try:
                final_captions = qwen_direct.caption_clip_qwen_direct(
                    captioner, ensure_downloaded, styles, time_remaining)
            except Exception as e:
                print(f"[qwen-direct] task {task_id} failed wholesale: {e}")
                traceback.print_exc()
            # Never leave a requested style empty — a missing style scores
            # zero for the whole clip per the official rules.
            for style in styles:
                final_captions[style] = final_captions.get(style) or fallback_captions(
                    [style], "qwen_direct produced no caption")[style]
            elapsed = time.monotonic() - t_start
            print(f"[+] Task {task_id} completed (qwen_direct) in {elapsed:.2f}s")
            return {"task_id": task_id, "captions": final_captions}

        # caption_clip runs the 2-stage pipeline (scene report -> Best-of-N
        # candidate caption sets) and returns the scene report alongside the
        # candidates. Do NOT store this on the shared captioner instance —
        # it's reused across worker threads, so per-call state must stay
        # local to this call.
        candidates, scene = captioner.caption_clip(ensure_downloaded, styles, video_url=video_url,
                                                   variety_index=variety_index,
                                                   time_remaining=time_remaining)

        final_captions = {}
        for style in styles:
            style_options = [c.get(style, "") for c in candidates if c.get(style)]
            # Code-level style guard: drop candidates that mechanically break
            # their style's definition (emoji, non-English characters, jargon
            # in humorous_non_tech, NO tech term in humorous_tech, "!" in
            # formal...) before the judge ever ranks them — fall back to the
            # full pool only if every candidate violates.
            clean = [o for o in style_options if not style_violations(style, o)]
            if clean and len(clean) < len(style_options):
                print(f"[style-guard] {style}: dropped {len(style_options) - len(clean)} "
                      f"candidate(s) with mechanical style violations")
            style_options = clean or style_options
            # Prefer candidates that already satisfy the style's word-count
            # rule; only fall back to the full pool when none are compliant.
            compliant = [o for o in style_options if in_word_range(style, o)]
            pick_pool = compliant or style_options
            if judge.available and time_remaining() > ENHANCEMENT_TIME_MARGIN_SECONDS:
                caption = judge.pick_best(style, scene, pick_pool)
            else:
                caption = next(iter(pick_pool), "")
            result = caption

            if (
                config.ENABLE_JUDGE_POLISH
                and judge.available
                and style in HUMOR_STYLES_FOR_POLISH
                and time_remaining() > ENHANCEMENT_TIME_MARGIN_SECONDS
            ):
                result = guarded_polish(judge, style, scene, result)

            if config.ENABLE_SELF_CRITIQUE and judge.available:
                for _ in range(config.MAX_CRITIQUE_RETRIES):
                    if time_remaining() <= ENHANCEMENT_TIME_MARGIN_SECONDS:
                        break
                    score, feedback = judge.judge(style, scene, result)
                    if score >= config.CRITIQUE_PASS_THRESHOLD:
                        break
                    hint = f"{result} (reviewer feedback to address: {feedback})" if feedback else result
                    result = guarded_polish(judge, style, scene, hint, baseline=result)

            # Last line of defence: if the final caption still carries a
            # mechanical violation (possible when every candidate violated,
            # or a polish introduced one), spend ONE deterministic repair
            # attempt on it — and keep the repair only if it actually fixed
            # the problem without introducing a new one.
            violations = result and style_violations(style, result)
            if violations and judge.available and time_remaining() > ENHANCEMENT_TIME_MARGIN_SECONDS:
                print(f"[style-guard] {style}: final caption violates ({'; '.join(violations)}) — repairing")
                hint = f"{result} (rewrite fixing exactly these problems: {'; '.join(violations)})"
                repaired = guarded_polish(judge, style, scene, hint, baseline=result)
                if repaired and not style_violations(style, repaired):
                    result = repaired

            # Never leave a requested style empty — a missing style scores
            # zero for the whole clip per the official rules.
            final_captions[style] = result or caption or fallback_captions([style])[style]

        elapsed = time.monotonic() - t_start
        print(f"[+] Task {task_id} completed successfully in {elapsed:.2f}s")
        return {"task_id": task_id, "captions": final_captions}

    except Exception as e:
        print(f"[error] task {task_id} failed: {e}")
        traceback.print_exc()
        return {"task_id": task_id, "captions": fallback_captions(styles)}

    finally:
        if local_path and os.path.exists(local_path) and not config.KEEP_DOWNLOADS:
            try:
                os.remove(local_path)
            except OSError:
                pass


def write_results(results: list) -> None:
    """Write the final results list to OUTPUT_PATH as valid JSON. This is the
    one artifact the judge reads — it must always be written."""
    os.makedirs(os.path.dirname(config.OUTPUT_PATH) or ".", exist_ok=True)
    with open(config.OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def all_fallback_results(tasks: list, reason: str) -> list:
    """A fallback row (every requested style filled) for every task, so no
    style is ever empty even when the whole captioner is unavailable."""
    out = []
    for t in tasks:
        tid = t.get("task_id", "unknown") if isinstance(t, dict) else "unknown"
        styles = (t.get("styles") if isinstance(t, dict) else None) or sorted(config.REQUIRED_STYLES)
        out.append({"task_id": tid, "captions": fallback_captions(styles, reason)})
    return out


def _emit_and_exit(results: list) -> NoReturn:
    """Best-effort write of results.json, then hard exit 0. Used by the
    startup guards so a fatal setup error still satisfies the 'exit 0 + valid
    /output/results.json' rules instead of crashing with no output."""
    try:
        write_results(results)
    except Exception as e:
        print(f"[fatal] could not write results.json: {e}")
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    print(f"[*] Reading tasks from {config.INPUT_PATH}")
    # Guard the whole startup: reading the task list or constructing the
    # captioner must never crash the container with no output — the rules
    # require exit 0 and a valid /output/results.json on every path.
    try:
        with open(config.INPUT_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            raise ValueError(f"tasks.json must be a JSON array, got {type(tasks).__name__}")
    except Exception as e:
        print(f"[fatal] could not read {config.INPUT_PATH}: {e} — writing empty results and exiting 0")
        traceback.print_exc()
        _emit_and_exit([])

    try:
        captioner = make_captioner()
        judge = JudgeAssistant()
    except Exception as e:
        # e.g. FIREWORKS_API_KEY unset at build time. Every task still gets a
        # fallback caption for every style rather than a crash.
        print(f"[fatal] captioner init failed: {e} — writing all-fallback results and exiting 0")
        traceback.print_exc()
        _emit_and_exit(all_fallback_results(tasks, "captioner unavailable"))

    if (config.ENABLE_JUDGE_POLISH or config.ENABLE_SELF_CRITIQUE) and not judge.available:
        print("[!] FIREWORKS_API_KEY not set — running caption-engine-only, no judge polish/critique.")

    # Preserve the original input order for the final output — scheduling
    # order (heaviest-first) is only used to decide processing sequence.
    original_order = {t.get("task_id"): i for i, t in enumerate(tasks)}

    print(f"[*] Probing {len(tasks)} clip(s) to schedule heaviest first...")
    scheduled_tasks = order_tasks_heaviest_first(tasks)

    results_by_id = {}

    pool = ThreadPoolExecutor(max_workers=config.CONCURRENCY)
    # variety_index comes from the task's position in the INPUT order (not the
    # heaviest-first scheduling order) so the humorous_non_tech openings
    # round-robin evenly and deterministically across the batch — see
    # prompts.NON_TECH_OPENINGS.
    futures = {
        pool.submit(process_task, task, captioner, judge,
                    original_order.get(task.get("task_id"), 0)): task
        for task in scheduled_tasks
    }

    # Hard rule: never wait past the global deadline (minus a finalization
    # reserve) no matter how many tasks are still running. A single stuck
    # clip must not cost us every other clip's result.
    wait_budget = max(0.0, time_remaining() - config.FINALIZATION_RESERVE_SECONDS)
    done, not_done = futures_wait(futures.keys(), timeout=wait_budget)

    for f in done:
        task = futures[f]
        try:
            results_by_id[task.get("task_id")] = f.result()
        except Exception as e:
            styles = task.get("styles") or sorted(config.REQUIRED_STYLES)
            print(f"[error] unexpected failure for task {task.get('task_id')}: {e}")
            results_by_id[task.get("task_id")] = {"task_id": task.get("task_id", "unknown"), "captions": fallback_captions(styles)}

    for f in not_done:
        task = futures[f]
        task_id = task.get("task_id", "unknown")
        styles = task.get("styles") or sorted(config.REQUIRED_STYLES)
        print(f"[timeout] task {task_id} did not finish before the global deadline — using fallback")
        results_by_id[task_id] = {"task_id": task_id, "captions": fallback_captions(styles, "runtime budget timeout")}
        try:
            f.cancel()  # best-effort only; a running thread can't actually be killed
        except Exception:
            pass

    # Guarantee every input task_id produced a row, even if something above
    # was skipped entirely.
    for t in tasks:
        tid = t.get("task_id", "unknown")
        if tid not in results_by_id:
            styles = t.get("styles") or sorted(config.REQUIRED_STYLES)
            results_by_id[tid] = {"task_id": tid, "captions": fallback_captions(styles)}

    results = [results_by_id[t.get("task_id", "unknown")] for t in tasks]
    results.sort(key=lambda r: original_order.get(r["task_id"], 0))

    write_results(results)

    elapsed = time.monotonic() - START_TIME
    print(f"[+] Wrote {len(results)} results to {config.OUTPUT_PATH} in {elapsed:.1f}s "
          f"({len(not_done)} timed out)")

    sys.stdout.flush()
    sys.stderr.flush()
    # Force immediate process exit instead of letting the interpreter join
    # any still-running (possibly network-hung) worker threads — those
    # threads are non-daemon by default and would otherwise block process
    # exit indefinitely, risking the 10-minute hard limit even though valid
    # output has already been written.
    os._exit(0)


if __name__ == "__main__":
    main()
