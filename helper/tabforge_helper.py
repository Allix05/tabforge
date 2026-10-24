"""TabForge local helper.

Runs on your own machine (free, nothing leaves it except the download itself):
  * serves the TabForge web app at http://127.0.0.1:8765
  * downloads audio from YouTube / SoundCloud / any yt-dlp site
  * resolves Spotify / Apple Music / Deezer links to the same song on YouTube
    (those services never expose their audio, so we look the track up by name)
  * optionally splits the song into stems with Demucs (htdemucs_6s: drums, bass,
    vocals, guitar, piano, other) so each tab mode hears only what it needs.

Usage:  python helper/tabforge_helper.py [--port 8765]
"""

import argparse
import html
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.parse
import urllib.request
import uuid
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent
WORK = Path(tempfile.gettempdir()) / "tabforge-jobs"
SR = 44100
STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]
COMBOS = {
    "instruments": ["guitar", "piano", "other"],
    "accomp": ["bass", "guitar", "piano", "other"],
}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"

jobs = {}
jobs_lock = threading.Lock()
separate_lock = threading.Lock()  # one Demucs run at a time keeps memory sane
_model = None


# ---------- optional dependencies ----------

def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def load_demucs():
    """Import Demucs without needing torchaudio: demucs only uses it for file
    I/O (we use ffmpeg + wave instead) and openunmix imports it at module load."""
    try:
        import torchaudio  # noqa: F401
    except Exception:
        sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    return torch, apply_model, get_model


def separation_available():
    try:
        load_demucs()
        return ffmpeg_exe() is not None
    except Exception:
        return False


def device_name():
    try:
        torch = load_demucs()[0]
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return None


# ---------- link resolution / download ----------

STREAMING_HOSTS = ("open.spotify.com", "spotify.link", "music.apple.com", "deezer.com", "deezer.page.link", "tidal.com")


def fetch_text(url, ua=UA):
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", "replace")


def resolve_streaming(url):
    """Return 'Artist - Title' for a streaming-service track page."""
    # streaming sites only render metadata server-side for link-preview bots
    page = fetch_text(url, ua="facebookexternalhit/1.1")

    def meta(prop):
        m = re.search(r'<meta[^>]+(?:property|name)="%s"[^>]+content="([^"]*)"' % re.escape(prop), page)
        return html.unescape(m.group(1)).strip() if m else ""

    title = meta("og:title")
    desc = meta("og:description")
    page_title = html.unescape((re.search(r"<title>(.*?)</title>", page, re.S) or [None, ""])[1]).strip()
    artist = meta("music:musician_description")
    m = re.match(r"(.+?) - (?:song|Song)[^|]*? by (.+?) \| Spotify", page_title)
    if m and not artist:
        title, artist = title or m.group(1), m.group(2)
    if not artist and desc:
        # Spotify: "Artist · Album · Song · 2020"; Apple: "Listen to Title by Artist on Apple Music"
        m = re.search(r"by (.+?) on Apple Music", desc)
        artist = m.group(1) if m else re.split(r" [·\-–] ", desc)[0].strip()
    if not title:
        raise RuntimeError("Could not read the track name from that link.")
    return f"{artist} - {title}".strip(" -")


def download(job, url):
    import yt_dlp

    target = url
    if any(h in url for h in STREAMING_HOSTS):
        job_update(job, message="Reading track info from the streaming link…")
        query = resolve_streaming(url)
        job_update(job, message=f"Finding “{query}” on YouTube…", title=query)
        target = f"ytsearch1:{query} official audio"

    out_dir = WORK / job["id"]

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            pct = (d.get("downloaded_bytes", 0) / total) if total else 0
            job_update(job, message=f"Downloading audio… {pct:.0%}", progress=pct)

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "max_filesize": 300 * 1024 * 1024,
    }
    if shutil.which("node"):
        # recent YouTube needs a JS runtime for signature solving; Node is common
        opts["js_runtimes"] = {"node": {}}
    ff = ffmpeg_exe()
    if ff:
        opts["ffmpeg_location"] = ff
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(target, download=True)
        if info.get("entries"):
            info = info["entries"][0]
    files = [p for p in out_dir.glob("source.*") if p.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("Download failed.")
    title = info.get("title") or job.get("title") or "Untitled"
    artist = info.get("artist") or info.get("uploader") or ""
    job_update(job, title=job.get("title") or title, artist=artist, source_url=info.get("webpage_url", url))
    return files[0]


# ---------- separation ----------

def decode(path):
    import numpy as np
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def write_wav(path, mono):
    import numpy as np
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    import numpy as np
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32767


def separate(job, src):
    global _model
    torch, apply_model, get_model = load_demucs()
    with separate_lock:
        job_update(job, status="separating", message="Loading the Demucs separation model…", progress=0)
        if _model is None:
            _model = get_model("htdemucs_6s")
            _model.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        wav = torch.from_numpy(decode(src))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        started = time.time()
        minutes = wav.shape[1] / SR / 60
        job_update(job, message=f"Separating vocals / guitar / piano / bass / drums on {dev.upper()} "
                                f"(≈{'20 s' if dev == 'cuda' else f'{max(1, round(minutes * 0.8))} min'})…")
        stop = threading.Event()

        def ticker():
            while not stop.wait(1):
                job_update(job, elapsed=round(time.time() - started))

        threading.Thread(target=ticker, daemon=True).start()
        try:
            with torch.no_grad():
                out = apply_model(_model, ((wav - mean) / std)[None], device=dev, split=True, overlap=0.25, progress=False)[0]
        finally:
            stop.set()
        out = out * std + mean
        stems = {}
        for name, stem in zip(_model.sources, out):
            path = WORK / job["id"] / f"{name}.wav"
            write_wav(path, stem.mean(0).numpy())
            stems[name] = path
        job_update(job, stems=sorted(stems) + sorted(COMBOS))


def stem_path(job, name):
    d = WORK / job["id"]
    if name == "mix":
        return job["file"]
    if name in STEMS:
        return d / f"{name}.wav"
    if name in COMBOS:
        p = d / f"{name}.wav"
        if not p.exists():
            parts = [read_wav(d / f"{s}.wav") for s in COMBOS[name]]
            write_wav(p, sum(parts))
        return p
    return None


# ---------- jobs ----------

def job_update(job, **kw):
    with jobs_lock:
        job.update(kw)


def public(job):
    keys = ("id", "status", "message", "progress", "title", "artist", "stems", "error", "elapsed", "source_url")
    return {k: job.get(k) for k in keys}


def run_job(job, url=None, upload=None, want_separation=False):
    try:
        if url:
            job_update(job, status="downloading", message="Starting download…")
            job["file"] = download(job, url)
        else:
            job["file"] = upload
        if want_separation:
            separate(job, job["file"])
        job_update(job, status="done", message="Ready", progress=1)
    except Exception as e:  # report anything to the browser instead of dying
        msg = str(e)
        if "Sign in to confirm" in msg:
            msg = "YouTube asked for a sign-in for this video. Try another upload of the song, or download the audio and drop the file in."
        job_update(job, status="error", error=msg, message="Failed")


def new_job():
    job = {"id": uuid.uuid4().hex[:12], "status": "queued", "message": "Queued", "progress": 0, "stems": [], "created": time.time()}
    (WORK / job["id"]).mkdir(parents=True, exist_ok=True)
    with jobs_lock:
        jobs[job["id"]] = job
    return job


# ---------- HTTP ----------

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        if "/api/jobs/" in (args[0] if args else "") and '" 200' in fmt % args:
            return  # don't spam the console with progress polls
        sys.stderr.write("  " + fmt % args + "\n")

    def end_headers(self):
        origin = self.headers.get("Origin", "")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        if self.path.endswith((".js", ".html", ".css")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/health":
            return self.send_json({"ok": True, "version": VERSION, "separation": SEPARATION, "device": DEVICE})
        m = re.fullmatch(r"/api/jobs/(\w+)", path)
        if m:
            job = jobs.get(m.group(1))
            return self.send_json(public(job)) if job else self.send_json({"error": "unknown job"}, 404)
        m = re.fullmatch(r"/api/jobs/(\w+)/audio/(\w+)", path)
        if m:
            job = jobs.get(m.group(1))
            if not job or job.get("status") != "done":
                return self.send_json({"error": "not ready"}, 404)
            p = stem_path(job, m.group(2))
            if not p or not Path(p).exists():
                return self.send_json({"error": "no such stem"}, 404)
            data = Path(p).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(p))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/jobs":
            return self.send_json({"error": "not found"}, 404)
        q = urllib.parse.parse_qs(parsed.query)
        want_sep = q.get("separate", ["0"])[0] == "1" and SEPARATION
        length = int(self.headers.get("Content-Length", 0))
        ctype = self.headers.get("Content-Type", "")
        if ctype.startswith("application/json"):
            url = json.loads(self.rfile.read(length) or b"{}").get("url", "").strip()
            if not re.match(r"https?://", url):
                return self.send_json({"error": "Please paste a full http(s) link."}, 400)
            # same link again (e.g. after switching modes): reuse the finished work
            with jobs_lock:
                for old in jobs.values():
                    if old.get("url") == url and old["status"] == "done" and (old.get("separated") or not want_sep):
                        return self.send_json(public(old), 201)
            job = new_job()
            job_update(job, url=url, separated=want_sep)
            threading.Thread(target=run_job, args=(job,), kwargs={"url": url, "want_separation": want_sep}, daemon=True).start()
        else:
            job = new_job()
            name = re.sub(r"[^\w.-]", "_", q.get("name", ["upload.bin"])[0])[-80:]
            dest = WORK / job["id"] / ("source" + (Path(name).suffix or ".bin"))
            with open(dest, "wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            job_update(job, title=Path(name).stem)
            threading.Thread(target=run_job, args=(job,), kwargs={"upload": dest, "want_separation": want_sep}, daemon=True).start()
        self.send_json(public(job), 201)


def main():
    global SEPARATION, DEVICE
    ap = argparse.ArgumentParser(description="TabForge local helper")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        import yt_dlp  # noqa: F401
        links = True
    except ImportError:
        links = False
    SEPARATION = separation_available()
    DEVICE = device_name() if SEPARATION else None
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  TabForge helper {VERSION}")
    print(f"  App:              {url}")
    print(f"  Links (yt-dlp):   {'yes' if links else 'NO  -> pip install yt-dlp'}")
    print(f"  Stem separation:  {('yes (' + DEVICE + ')') if SEPARATION else 'no  -> pip install -r helper/requirements-separation.txt'}")
    print("  Press Ctrl+C to stop.\n")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if not args.no_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


SEPARATION = False
DEVICE = None

if __name__ == "__main__":
    main()
