# CaptionForge

Track 2 (Video Captioning Agent) submission for **AMD Developer Hackathon: Act II**.

An AI agent that watches a video clip and generates captions in the four
requested styles (`formal`, `sarcastic`, `humorous_tech`,
`humorous_non_tech`), built on Fireworks-hosted models with deterministic
in-code style and robustness guards.

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
task's `styles` list requests. Container must exit 0, must be ready within
60s, and the whole run must finish within **10 minutes** total. The
pipeline guarantees a caption for every requested style on every path
(fallbacks instead of crashes), a hard global time budget with
heaviest-clip-first scheduling, and exit 0 with valid JSON output even
when setup itself fails.

## Credentials: Track 2 injects NONE — must be baked into the image

The official guide states Track 2 injects **no** API key or model
restriction at evaluation time: *"use your own credentials inside the
container."* The judge runs `docker run <image>` with no `-e` flags, so
`FIREWORKS_API_KEY` must be baked in at **build time** via `--build-arg`.

Never put the real key literally in the `Dockerfile` or commit it to git —
only pass it on the build command line. The pushed public image will have
the key embedded in its layers (extractable by anyone who pulls it): treat
this hackathon key as disposable, watch the credit balance during the
event, and rotate/revoke it after the event ends. Any other credentials in
`.env` (keys used only by local dev scripts) stay local and are never baked.

## Local dev

```bash
cp .env.example .env   # fill in FIREWORKS_API_KEY
pip install -r requirements.txt
python src/main.py      # reads input/tasks.json, writes output/results.json
```

Set `KEEP_DOWNLOADS=true` locally to cache clips in `scratch_videos/`
between runs. On home bandwidth, also use `CONCURRENCY=1` and raise
`TOTAL_BUDGET_SECONDS` / `MAX_DOWNLOAD_WALL_SECONDS` — the defaults are
tuned for the judging VM's datacenter bandwidth.

## Web demo (optional)

`web_demo/` is a small Flask app for trying the pipeline interactively — not
part of the submission (the real container is headless).

```bash
pip install -r web_demo/requirements.txt
python web_demo/app.py   # http://localhost:5000
```

## Docker — build, test, and push

Submission requires an image pushed to a public registry. Judging VM is
`linux/amd64`.

```bash
# Local build/test (key via build-arg, then a harness-style run with no -e)
docker build --build-arg FIREWORKS_API_KEY=$FIREWORKS_API_KEY -t captionforge .
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  captionforge

# REAL SUBMISSION BUILD — self-contained image (the judge passes no -e flags):
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

- `src/main.py` — orchestration, concurrency, time budget, startup guards, fallbacks
- `src/qwen_direct.py` — primary caption engine
- `src/fireworks_vision_client.py` — Fireworks client + ffmpeg frame extraction
- `src/judge_polish.py` — legacy engine helpers
- `src/prompts.py` — prompt construction + deterministic style checks
- `src/downloader.py` — clip download with retry/timeout/wall-cap + size probing
- `src/config.py` — all tunables, env-var driven
- `scripts/` — local dev/eval tooling (never in the image)
- `web_demo/` — Flask demo UI (not part of the submission)
- `Dockerfile` — the Track 2 submission image (headless, key baked at build time)
- `Dockerfile.web` — the hosted demo image (web server, key supplied at run time)
- `docs/` — the official Participant Guide
