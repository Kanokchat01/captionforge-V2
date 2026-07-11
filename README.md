# CaptionForge

Track 2 (Video Captioning Agent) submission for **AMD Developer Hackathon: Act II**.

Watches a video clip and writes one caption per requested style —
`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech` — using a
three-model Fireworks pipeline where each model does the job it won in a
head-to-head benchmark (2026-07-11, official sample clips, cross-judged on
the official rubric by two neutral models).

**Contents:** [Model pipeline](#model-pipeline) · [Accuracy & style safeguards](#accuracy--style-safeguards) ·
[Official I/O contract](#official-io-contract-do-not-change) · [Pipeline stages](#pipeline-stages) ·
[Credentials](#credentials-track-2-injects-none--must-be-baked-into-the-image) ·
[Local dev](#local-dev) · [Web demo](#web-demo-optional) · [Docker](#docker--build-test-and-push) ·
[Project layout](#project-layout)

## Model pipeline

| Role | Model | Why |
|---|---|---|
| Stage 1: scene analysis (native video) | `minimax-m3` | the only account model that accepts the WHOLE clip via `video_url` (verified by real calls; kimi/qwen reject video input) — sees motion, real timelines, and camera movement that still frames can't, in ~4–12s per clip with no local download at all |
| Stage 1 fallback #1 (frames) | `kimi-k2p7-code` | most detailed, meme-aware frame-based scene reports; verified hallucination-free against real frames |
| Stage 1 fallback #2 (frames) | `qwen3p7-plus` | vision-capable second opinion if kimi also fails on a clip |
| Stage 2: caption writing | `glm-5p2` | best caption writer (0.874 vs 0.850 qwen3p7-plus, 0.830 kimi-k2p7-code, 0.666 minimax-m3) |
| Best-of-N pick / judge / polish | `qwen3p7-plus` | runner-up quality, fastest, different family from the writer (no self-preference bias) |

`minimax-m3` is deliberately **only** the eyes, never the writer: it scored
0.666 on caption writing and failed JSON output on 2 of 3 benchmark clips —
but Stage 1 returns plain text, so neither weakness applies there. Its OCR
is also untrusted (it confidently misread a real building sign in testing),
which is why the prompts ban transcribing on-screen text outright.

## Accuracy & style safeguards

The scene-report and caption prompts (`src/prompts.py`) are hardened against
the two most common ways a vision-language model loses accuracy points on
this task:

- **No guessed specifics.** The model is explicitly forbidden from naming a
  real-world city/country/landmark or capture equipment ("drone shot",
  lens/exposure) — it must describe generically instead ("a multi-lane
  urban street"). Playback effects the motion itself proves (time-lapse,
  slow motion) are allowed only on the native-video path, where they are
  actually observable.
- **On-screen text is NEVER transcribed.** Compressed video makes model OCR
  confidently wrong (in testing it misread a real "KOREA ILLIES
  ENGINEERING" sign as "KOREA MEDIA ENGINEERING"), and a caption quoting
  misread text is a guaranteed accuracy penalty while a generic "a building
  sign" never is — so both the report and caption prompts ban quoting any
  sign/label/screen wording outright, no legibility exception.
- **No upgraded poses.** An ambiguous pose is reported as the literal,
  observable position ("looking down"), not promoted into a stronger claim
  ("eyes closed", "falling asleep", "about to pounce") unless a frame shows
  it beyond doubt.
- **Sarcastic stays dry.** Tuned to the official spec ("dry, ironic, lightly
  mocking") — deadpan understatement, not hype slang.

On top of the prompts, `src/main.py` enforces two rules in code (zero extra
API cost):

- **Word-count guard** — Best-of-N candidates are pre-filtered to the
  style's required word range before the judge picks; a self-critique polish
  that would push an in-range caption *out* of range is automatically
  reverted (`guarded_polish`).
- **Style-conformant fallbacks** — if a clip fails, every style still gets a
  generic caption that obeys its own word-count/emoji rules
  (`FALLBACK_CAPTION_BY_STYLE`), instead of an "unavailable" message that
  would score zero on both accuracy and style match.

## Official I/O contract (do not change)

Reads `/input/tasks.json`:
```json
[{"task_id": "v1", "video_url": "https://...", "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}]
```

Writes `/output/results.json`:
```json
[{"task_id": "v1", "captions": {"formal": "...", "sarcastic": "...", "humorous_tech": "...", "humorous_non_tech": "..."}}]
```

Style keys use underscores, not hyphens — must match exactly what each
task's `styles` list requests, or that clip scores zero for the missing
style. Container must exit 0, must be ready within 60s, and the whole run
must finish within **10 minutes** total (hidden eval set is ~12 clips,
30s–2min each). Scoring is caption **accuracy** (0–1) + **style match**
(0–1) per caption by an LLM judge — there is no token-count penalty in
Track 2, which is why the pipeline spends extra calls on quality
(Best-of-N + self-critique).

## Pipeline stages

1. **Stage 1 — Scene Report, native video** (`minimax-m3`): the task's
   `video_url` goes straight to Fireworks (fetched server-side — no local
   download at all) and the model watches the whole clip, producing a
   structured 10-section report (subject, environment, timeline, camera,
   lighting, audio, mood, standout details, humor potential, RISKS) that
   includes real motion, continuous timelines, and camera movement.
2. **Download — only if the frame fallback is needed** (lazy): retry,
   `.tmp`-then-rename, and a hard wall-clock cap per download
   (`MAX_DOWNLOAD_WALL_SECONDS`, default 150s) so a slow trickling server
   can't eat the global budget.
3. **Stage 1 fallback — frames** (`kimi-k2p7-code`): adaptive frame sampling —
   one frame per ~8s, clamped to 8–16 frames, downscaled to 768px — same
   10-section report. Degrade chain: fewer frames → `qwen3p7-plus`.
4. **Stage 2 — Best-of-N captions** (`glm-5p2`): N=5 candidate caption sets
   generated in parallel at temperatures 0.55/0.7/0.85/1.0/1.15, text-only
   from the report. Style prompts lead with the official Participant Guide
   definitions verbatim; word ranges and openings are soft craft guidance,
   not straitjackets.
5. **Judge pass** (`qwen3p7-plus`): prefers word-range-compliant candidates,
   picks the best per style on the contest's two official axes (accuracy +
   style match, with humor sharpness as the tiebreaker for humor styles),
   then self-critiques — any caption scoring below
   `CRITIQUE_PASS_THRESHOLD` (8/10) is rewritten with the judge's feedback,
   up to `MAX_CRITIQUE_RETRIES` (2) rounds, reverting if the rewrite breaks
   the word-count guard.
6. **Time budget**: clips are probed (HEAD) and scheduled heaviest-first,
   processed with `CONCURRENCY=6`; a hard wall-clock deadline
   (`TOTAL_BUDGET_SECONDS=540`) is enforced with
   `concurrent.futures.wait(..., timeout=...)` — any clip not done in time
   gets a fallback caption instead of blocking the rest. Startup itself is
   guarded too (malformed `tasks.json` or a missing API key still produces a
   valid `results.json` and exit 0, never a crash with no output).
   `os._exit(0)` at the end guarantees the process can't hang on a stuck
   thread.

No audio understanding on any path — Fireworks' Whisper endpoints were
discontinued 2026-06-10, and `video_url` input delivers no audio track to
the model (verified); prompts explicitly force "No audio present" so the
models never invent sound.

## Credentials: Track 2 injects NONE — must be baked into the image

The official guide states Track 2 injects **no** API key or model
restriction at evaluation time: *"use your own credentials inside the
container."* The judge just runs `docker run <image>` with no `-e` flags,
so `FIREWORKS_API_KEY` must be baked in at **build time** via
`--build-arg`.

Never put the real key literally in the `Dockerfile` or commit it to git —
only pass it on the build command line. The pushed public image will have
the key embedded in its layers (extractable by anyone who pulls it): treat
this hackathon key as disposable, watch the credit balance during the
event, and rotate/revoke it after the event ends.

## Local dev

```bash
cp .env.example .env   # fill in FIREWORKS_API_KEY
pip install -r requirements.txt
python src/main.py      # reads input/tasks.json, writes output/results.json
```

Set `KEEP_DOWNLOADS=true` locally to cache clips in `scratch_videos/`
between runs. Note: a full 12-clip UHD run on home bandwidth will hit the
download caps by design — the judging VM has datacenter bandwidth.

## Web demo (optional)

`web_demo/` is a small Flask app for trying the pipeline interactively — not
part of the submission (the real container is headless). It reuses the exact
same pipeline code as `src/main.py`, so results match what the judged
container would produce for the same clip.

```bash
pip install -r web_demo/requirements.txt
python web_demo/app.py   # http://localhost:5000
```

Paste a clip URL (or click one of the three official example clips), pick
styles, and watch live progress stream in — native-video scene report (the
download step is skipped unless the frame fallback runs) → Best-of-N
candidates → judge pick & self-critique — with each caption's word-count
compliance shown as it arrives.

## Docker — build, test, and push

Submission requires an actual image pushed to a public registry. Judging VM
is `linux/amd64`.

```bash
# Local build/test
docker build -t captionforge .
docker run --rm \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  captionforge

# REAL SUBMISSION BUILD — bake the credential in via --build-arg so the
# image is self-contained (the judge passes no -e flags):
docker buildx build --platform linux/amd64 \
  --build-arg FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -t ghcr.io/<you>/captionforge:latest --push .

# Sanity-check with ZERO -e flags, exactly like the judge will run it:
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  ghcr.io/<you>/captionforge:latest
```

Submissions are rate-limited to 10/hour — only submit after a clean local
Docker run. `.dockerignore` keeps the build context to just
`requirements.txt` + `src/` (docs, the web demo, `.venv`, and `.env` never
reach the image).

## Project layout

- `src/main.py` — orchestration, concurrency, time budget, startup guards,
  fallback handling, word-count guard
- `src/fireworks_vision_client.py` — Stage 1 vision + Stage 2 Best-of-N generation
- `src/judge_polish.py` — judge: pick-best, critique, polish (model set by `FIREWORKS_JUDGE_MODEL`)
- `src/prompts.py` — all prompt text, style rules, judge rubrics, accuracy safeguards
- `src/downloader.py` — clip download with retry/timeout/wall-cap + size probing
- `src/config.py` — all tunables, env-var driven, benchmark notes
- `web_demo/` — Flask demo with live-progress streaming UI (not part of the submission)
- `Dockerfile` — the Track 2 submission image (headless, key baked at build time)
- `Dockerfile.web` — the hosted demo image (web server, key supplied at run time)
- `docs/` — the official Participant Guide
