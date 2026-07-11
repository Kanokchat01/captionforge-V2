"""
Download a task's video_url to local disk before frame extraction/captioning.
Bounded timeout + retries per the team's own error-handling plan (Use Case 1
in hackathon_strategy_guide.md): don't let one bad clip stall the whole run.

Also supports local file paths / file:// URIs directly (dev/testing
convenience only — the real judged input is always a real http(s) URL per
the official contract). This lets the team point tasks.json at video files
already sitting on disk without needing to spin up a local HTTP server just
to test them.
"""
import os
import shutil
import time
import urllib.parse

import requests

DOWNLOAD_TIMEOUT_SECONDS = 30   # connect + between-chunks read timeout
MAX_RETRIES = 3
PROBE_TIMEOUT_SECONDS = 8
# Hard wall-clock cap per download attempt. The requests/socket timeouts only
# catch a *stalled* connection — a server that keeps trickling bytes slowly
# passes those checks forever and can eat the whole 10-minute budget (this is
# exactly what killed the 12-clip local run; see time_bottleneck_analysis.md).
MAX_DOWNLOAD_WALL_SECONDS = float(os.environ.get("MAX_DOWNLOAD_WALL_SECONDS", "150"))


def _local_path_if_any(url: str):
    """Returns a filesystem path if `url` refers to a local file (a
    file:// URI, or a plain path that already exists on disk), else None.
    Real submissions always get http(s) URLs, so this only ever triggers
    during local dev/testing with tasks.json pointed at files on disk."""
    if url.startswith("file://"):
        parsed = urllib.parse.urlparse(url)
        candidate = urllib.parse.unquote(parsed.path)
        # On Windows, urlparse leaves a leading "/" in front of the drive
        # letter (e.g. "/C:/videos/x.mp4") — strip it so it's a valid path.
        if os.name == "nt" and len(candidate) > 2 and candidate[0] == "/" and candidate[2] == ":":
            candidate = candidate[1:]
        return candidate if os.path.exists(candidate) else None
    if not url.startswith("http://") and not url.startswith("https://") and os.path.exists(url):
        return url
    return None


def probe_size_mb(url: str) -> float:
    """Cheap HEAD request to estimate clip weight before downloading, so the
    scheduler can process heavier clips first (avoids a big 4K clip getting
    stuck at the back of the queue right as the time budget runs out).
    Returns 0.0 if the size can't be determined — caller should treat that as
    'unknown weight', not 'zero cost'."""
    local = _local_path_if_any(url)
    if local:
        try:
            return os.path.getsize(local) / (1024 * 1024)
        except OSError:
            return 0.0
    try:
        resp = requests.head(url, timeout=PROBE_TIMEOUT_SECONDS, allow_redirects=True)
        size = int(resp.headers.get("Content-Length", 0))
        return size / (1024 * 1024)
    except Exception:
        return 0.0


def download_video(url: str, output_dir: str = "/tmp/captionforge_videos") -> str:
    # Use workspace folder for local downloads on Windows so it doesn't try to write to root /tmp
    if os.name == "nt" and output_dir.startswith("/tmp"):
        output_dir = "scratch_videos"
        
    os.makedirs(output_dir, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0] or "clip.mp4"
    local_path = os.path.join(output_dir, filename)
    tmp_path = local_path + ".tmp"

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"[download] Using cached file: {local_path} ({file_size:.2f} MB)")
        return local_path

    # Local file / file:// URI — copy into the working dir rather than
    # returning the original path directly
    source_path = _local_path_if_any(url)
    if source_path:
        print(f"[download] Copying local file {source_path} to {local_path}...")
        t0 = time.monotonic()
        shutil.copy2(source_path, local_path)
        if os.path.getsize(local_path) == 0:
            raise IOError(f"Copied file is empty: {source_path}")
        elapsed = time.monotonic() - t0
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"[download] Copied {filename} ({file_size:.2f} MB) in {elapsed:.2f}s")
        return local_path

    print(f"[download] Starting download of {url}...")
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.monotonic()
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            resp = requests.get(url, stream=True, timeout=(10, DOWNLOAD_TIMEOUT_SECONDS))
            resp.raise_for_status()

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
                    if time.monotonic() - t0 > MAX_DOWNLOAD_WALL_SECONDS:
                        raise TimeoutError(
                            f"download exceeded wall-clock cap of {MAX_DOWNLOAD_WALL_SECONDS:.0f}s "
                            f"({os.path.getsize(tmp_path) / (1024 * 1024):.1f} MB so far)")

            if os.path.getsize(tmp_path) == 0:
                raise IOError("Downloaded file is empty")
                
            # Rename only on successful complete download
            if os.path.exists(local_path):
                os.remove(local_path)
            os.rename(tmp_path, local_path)
            
            elapsed = time.monotonic() - t0
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            rate = file_size / elapsed if elapsed > 0 else 0
            print(f"[download] Success: {filename} ({file_size:.2f} MB) in {elapsed:.2f}s ({rate:.2f} MB/s)")
            return local_path
        except Exception as e:
            last_error = e
            print(f"[download] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            # A wall-clock cap hit means the link is just slow, not flaky —
            # retrying would burn another cap's worth of the global budget.
            if isinstance(e, TimeoutError):
                break
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts: {last_error}")
