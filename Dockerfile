# Build for the judging VM's architecture (linux/amd64). If building on
# Apple Silicon: docker buildx build --platform linux/amd64 -t <tag> --push .
FROM python:3.11-slim

# ffmpeg/ffprobe are required by the frame-extraction pipeline
# (fireworks_vision_client.py).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
WORKDIR /app/src

# --- Track 2 credential baking ---
# The official Participant Guide states that Track 2 injects NO API key at
# evaluation time ("use your own credentials inside the container") — unlike
# Track 1, the judge just runs `docker run <image>` with no -e flags. So our
# own credentials must be baked into the image itself at build time via
# --build-arg, not supplied at `docker run` time. Never put the real key
# literally in this file or commit it — only pass it on the build command
# line (see README.md for the exact submission build command).
# ONLY the Fireworks key goes in. OPENROUTER_API_KEY is a local-eval-only
# credential and must NEVER be baked: this image is publicly pullable.
ARG FIREWORKS_API_KEY=""
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

# --- v6 engine knobs (baked per submission rung; empty string = default) ---
ARG CAPTION_ASSEMBLY="qwen_direct"
ARG QWEN_DIRECT_GUARD_LEVEL="1"
ARG QWEN_DIRECT_FRAMES="4"
ARG QWEN_DIRECT_TEMP_FORMAL=""
ARG QWEN_DIRECT_TEMP_SARCASTIC=""
ARG QWEN_DIRECT_TEMP_HUMOROUS_TECH=""
ARG QWEN_DIRECT_TEMP_HUMOROUS_NON_TECH=""
ENV CAPTION_ASSEMBLY=${CAPTION_ASSEMBLY} \
    QWEN_DIRECT_GUARD_LEVEL=${QWEN_DIRECT_GUARD_LEVEL} \
    QWEN_DIRECT_FRAMES=${QWEN_DIRECT_FRAMES} \
    QWEN_DIRECT_TEMP_FORMAL=${QWEN_DIRECT_TEMP_FORMAL} \
    QWEN_DIRECT_TEMP_SARCASTIC=${QWEN_DIRECT_TEMP_SARCASTIC} \
    QWEN_DIRECT_TEMP_HUMOROUS_TECH=${QWEN_DIRECT_TEMP_HUMOROUS_TECH} \
    QWEN_DIRECT_TEMP_HUMOROUS_NON_TECH=${QWEN_DIRECT_TEMP_HUMOROUS_NON_TECH}

# 24 parallel vision calls (6 clips x 4 styles) risk 429 clustering; 4x4=16
# keeps well under it and the engine is fast enough (~10s/clip) that the
# 540s budget still clears with huge headroom.
ENV CONCURRENCY="4"

# No CLI args: the harness runs the container, main.py reads /input/tasks.json
# and writes /output/results.json on its own, then exits 0.
CMD ["python", "main.py"]
