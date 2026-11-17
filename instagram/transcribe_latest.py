"""For every account in the following JSON: find their latest post, download its
audio, and transcribe it locally with Whisper.

How it works:
  1. GET https://www.instagram.com/<username>/ (logged out, browser-like headers).
     The HTML embeds the 12 most recent posts under
     xig_user_by_username.polaris_ordered_timeline_connection.
     Pinned posts are listed first, so "latest" = the node with the highest pk.
  2. yt-dlp downloads the audio track of /reel/<code>/ or /p/<code>/.
  3. asr.py transcribes it: Whisper detects the language, Hindi goes to Maya
     Transcribe, everything else stays on Whisper (--translate: Whisper, English).

Results are written to data/latest_posts.json and to MongoDB (social_posts, see
db.py). The run is resumable: already-transcribed posts are skipped, except
Hindi posts transcribed by Whisper before Maya was added, which are redone.

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

import os

import yt_dlp

import asr
import db
import env  # noqa: F401  loads the root .env into os.environ

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


COOKIE_FILE = DATA / "cookies.txt"


def write_cookie_file():
    """Instagram needs a login to download reels: hand yt-dlp the .env session
    as a Netscape cookie file (in data/, gitignored, readable only by you)."""
    cookies = {
        "sessionid": os.environ.get("IG_SESSIONID"),
        "csrftoken": os.environ.get("IG_CSRFTOKEN"),
        "ds_user_id": os.environ.get("IG_DS_USER_ID"),
    }
    if not cookies["sessionid"]:
        print("warning: IG_SESSIONID not set in .env; reel downloads will likely fail", file=sys.stderr)
        return None
    DATA.mkdir(parents=True, exist_ok=True)
    expiry = int(time.time()) + 30 * 86400
    lines = ["# Netscape HTTP Cookie File"]
    lines += [f".instagram.com\tTRUE\t/\tTRUE\t{expiry}\t{k}\t{v}" for k, v in cookies.items() if v]
    COOKIE_FILE.write_text("\n".join(lines) + "\n")
    COOKIE_FILE.chmod(0o600)
    return str(COOKIE_FILE)


def download_audio(url, stem):
    """Returns (list of audio file paths, timestamp). Images yield no files."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    opts = {
        "cookiefile": write_cookie_file(),
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


def needs_maya_redo(rec):
    """Hindi posts transcribed by Whisper before Maya was added."""
    return any(
        t.get("language") == "hi" and not str(t.get("model", "")).startswith("maya")
        for t in rec.get("transcripts") or []
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--following", default=str(next(DATA.glob("following_*.json"), "")),
                    help="following JSON from fetch_following.py")
    ap.add_argument("--model", default="large-v3-turbo", help="whisper model: tiny/base/small/medium/large-v3-turbo")
    ap.add_argument("--translate", action="store_true", help="translate speech to English")
    ap.add_argument("--device", default="cpu", help="whisper device: cpu or cuda (Maya picks the Mac GPU itself)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-audio", action="store_true", help="keep downloaded audio files")
    args = ap.parse_args()

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
            # retry posts whose download failed earlier (they have an empty transcript list)
            done = prev.get("transcripts") or prev.get("no_audio") or prev.get("media_type") == "image"
            if prev.get("code") == node["code"] and done and not needs_maya_redo(prev):
                print(f"  {node['code']} already done, skipping")
                db.save_post(prev)  # make sure it's in the DB too
                continue

            url = post_url(node)
            rec = {
                "username": name,
                "full_name": u.get("full_name"),
                "code": node["code"],
                "pk": str(node.get("pk")),
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
                t = asr.transcribe(f, args.model, args.translate, args.device)
                if not args.keep_audio:
                    Path(f).unlink(missing_ok=True)
                if t is None:
                    continue  # silent clip
                rec["transcripts"].append(t)
                print(f"  {node['code']} [{t['language']}, {t['duration']}s, {t['model']}] {t['text'][:100]}")
            if files and not rec["transcripts"]:
                rec["no_audio"] = True  # downloaded fine but silent: don't retry next run
            if not rec["transcripts"]:
                print(f"  {node['code']} has no audio ({rec['media_type']}); caption saved")
            rec["transcript"] = "\n\n".join(t["text"] for t in rec["transcripts"])
            results[name] = rec
            db.save_post(rec)
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
