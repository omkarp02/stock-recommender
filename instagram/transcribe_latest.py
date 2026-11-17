"""For every account in the following JSON: find their latest post, download its
audio, and transcribe it locally with Whisper.

How it works:
  1. GET https://www.instagram.com/<username>/ (logged out, browser-like headers).
     The HTML embeds the 12 most recent posts under
     xig_user_by_username.polaris_ordered_timeline_connection.
     Pinned posts are listed first, so "latest" = the node with the highest pk.
  2. yt-dlp downloads the audio track of /reel/<code>/ or /p/<code>/.
  3. faster-whisper transcribes it (auto language detection; --translate for English).

Results are written to data/latest_posts.json and the run is resumable:
already-transcribed posts are skipped.

Usage:
  .venv/bin/python transcribe_latest.py                 # all users
  .venv/bin/python transcribe_latest.py --limit 3       # first 3 users
  .venv/bin/python transcribe_latest.py --model small   # faster, less accurate
  .venv/bin/python transcribe_latest.py --translate     # English translation
"""
import argparse
import html as htmllib
import json
import random
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp

HERE = Path(__file__).parent
DATA = HERE / "data"
MEDIA = DATA / "media"
OUT = DATA / "latest_posts.json"

PAGE_HEADERS = {
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36",
    "accept": "text/html,application/xhtml+xml",
    "accept-language": "en-US,en;q=0.9",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
}
SCRIPT_RE = re.compile(r'<script type="application/json"[^>]*>(.*?)</script>', re.S)
MEDIA_TYPES = {1: "image", 2: "video", 8: "carousel"}


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = obj.values()
    if isinstance(obj, (list, tuple, type({}.values()))):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def latest_post(username):
    req = urllib.request.Request(f"https://www.instagram.com/{username}/", headers=PAGE_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        page = r.read().decode("utf-8", "replace")
    for raw in SCRIPT_RE.findall(page):
        if "polaris_ordered_timeline_connection" not in raw:
            continue
        conn = find_key(json.loads(htmllib.unescape(raw) if raw.startswith("&") else raw),
                        "polaris_ordered_timeline_connection")
        nodes = [e["node"] for e in (conn or {}).get("edges", [])]
        if nodes:
            return max(nodes, key=lambda n: int(n["pk"]))  # skips pinned older posts
    return None


def post_url(node):
    kind = "reel" if node.get("product_type") == "clips" else "p"
    return f"https://www.instagram.com/{kind}/{node['code']}/"


def download_audio(url, stem):
    """Returns (list of audio file paths, timestamp). Images yield no files."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(MEDIA / f"{stem}_%(autonumber)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if not info:
        return [], None
    entries = info.get("entries") or [info]
    files = []
    for e in entries:
        # skip images inside carousels: only entries that carry audio/video streams
        if not e or (e.get("acodec") in (None, "none") and e.get("vcodec") in (None, "none")):
            continue
        files += [d["filepath"] for d in e.get("requested_downloads") or [] if d.get("filepath")]
    ts = info.get("timestamp") or next((e.get("timestamp") for e in entries if e), None)
    return [f for f in files if Path(f).exists()], ts


_model = None
DEVICE = "cpu"


def transcribe(path, model_name, translate):
    global _model
    if _model is None:
        import av
        from faster_whisper import WhisperModel

        # faster-whisper 1.2.1 passes metadata_errors=, which PyAV >= 15 removed
        _open = av.open
        av.open = lambda *a, metadata_errors=None, **k: _open(*a, **k)
        print(f"loading whisper model '{model_name}' (first run downloads it)...")
        _model = WhisperModel(model_name, device=DEVICE, compute_type="int8" if DEVICE == "cpu" else "default")
    segments, info = _model.transcribe(
        path, task="translate" if translate else "transcribe", vad_filter=True, beam_size=5
    )
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "text": " ".join(s["text"] for s in segs),
        "segments": segs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--following", default=str(next(DATA.glob("following_*.json"), "")),
                    help="following JSON from fetch_following.py")
    ap.add_argument("--model", default="large-v3-turbo", help="whisper model: tiny/base/small/medium/large-v3-turbo")
    ap.add_argument("--translate", action="store_true", help="translate speech to English")
    ap.add_argument("--device", default="cpu", help="cpu, or mps (Apple GPU, if your ctranslate2 supports it)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-audio", action="store_true", help="keep downloaded audio files")
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device

    users = json.loads(Path(args.following).read_text())["users"]
    if args.limit:
        users = users[: args.limit]
    results = json.loads(OUT.read_text()) if OUT.exists() else {}

    for i, u in enumerate(users, 1):
        name = u["username"]
        print(f"[{i}/{len(users)}] {name}")
        try:
            node = latest_post(name)
            if not node:
                results[name] = {"username": name, "error": "no posts visible (private or blocked)"}
                print("  no posts visible")
                continue
            prev = results.get(name, {})
            if prev.get("code") == node["code"] and prev.get("transcripts") is not None:
                print(f"  {node['code']} already done, skipping")
                continue

            url = post_url(node)
            rec = {
                "username": name,
                "full_name": u.get("full_name"),
                "code": node["code"],
                "url": url,
                "media_type": MEDIA_TYPES.get(node.get("media_type"), node.get("media_type")),
                "product_type": node.get("product_type"),
                "caption": (node.get("caption") or {}).get("text"),
                "thumbnail": node.get("display_uri"),
            }
            files, ts = download_audio(url, f"{name}_{node['code']}")
            rec["posted_at"] = datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else None
            rec["transcripts"] = []
            for f in files:
                t = transcribe(f, args.model, args.translate)
                rec["transcripts"].append(t)
                print(f"  {node['code']} [{t['language']}, {t['duration']}s] {t['text'][:100]}")
                if not args.keep_audio:
                    Path(f).unlink(missing_ok=True)
            if not files:
                print(f"  {node['code']} has no audio ({rec['media_type']}); caption saved")
            rec["transcript"] = "\n\n".join(t["text"] for t in rec["transcripts"])
            results[name] = rec
        except Exception as e:  # keep going on per-user failures
            print(f"  error: {e}", file=sys.stderr)
            results[name] = {"username": name, "error": str(e)}
        finally:
            DATA.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        time.sleep(random.uniform(2, 4))

    ok = sum(1 for r in results.values() if r.get("transcripts"))
    print(f"done: {ok} with transcripts, {len(results)} total -> {OUT}")


if __name__ == "__main__":
    main()
