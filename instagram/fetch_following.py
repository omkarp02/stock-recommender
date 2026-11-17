"""Fetch the full Instagram "following" list for a user and save it as JSON.

Session values are read from env vars so they never land in the repo:
  IG_SESSIONID, IG_CSRFTOKEN, IG_DS_USER_ID  (required)
  IG_TARGET_USER_ID  (optional, defaults to IG_DS_USER_ID)

Usage:
  IG_SESSIONID=... IG_CSRFTOKEN=... IG_DS_USER_ID=... python3 instagram/fetch_following.py
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SESSIONID = os.environ["IG_SESSIONID"]
CSRF = os.environ["IG_CSRFTOKEN"]
DS_USER_ID = os.environ["IG_DS_USER_ID"]
TARGET = os.environ.get("IG_TARGET_USER_ID", DS_USER_ID)
OUT = Path(__file__).parent / "data" / f"following_{TARGET}.json"

HEADERS = {
    "accept": "*/*",
    "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1",
    "x-csrftoken": CSRF,
    "x-ig-app-id": "936619743392459",
    "x-requested-with": "XMLHttpRequest",
    "referer": "https://www.instagram.com/",
    "cookie": f"sessionid={SESSIONID}; csrftoken={CSRF}; ds_user_id={DS_USER_ID}",
}


def fetch_page(max_id):
    params = {"count": 100}
    if max_id:
        params["max_id"] = max_id
    url = f"https://www.instagram.com/api/v1/friendships/{TARGET}/following/?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 4:
                wait = 30 * (attempt + 1)
                print(f"  HTTP {e.code}, backing off {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise


def main():
    users, seen, max_id, page = [], set(), None, 0
    while True:
        data = fetch_page(max_id)
        for u in data.get("users", []):
            if u["pk"] not in seen:
                seen.add(u["pk"])
                users.append(u)
        page += 1
        print(f"page {page}: total {len(users)}")
        max_id = data.get("next_max_id")
        if not data.get("has_more") or not max_id:
            break
        time.sleep(random.uniform(1.5, 3.5))  # stay polite to avoid rate limits

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "target_user_id": TARGET,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(users),
        "users": users,
    }, indent=2, ensure_ascii=False))
    print(f"saved {len(users)} users -> {OUT}")


if __name__ == "__main__":
    main()
