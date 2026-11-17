"""MongoDB storage for scraped social data (DB_URL in the root .env).

Collections are platform-agnostic so other sources can share them later:

  social_accounts  one doc per followed creator      unique: (platform, user_id)
  social_posts     one doc per post, with transcript  unique: (platform, post_id)

Writes are upserts, so re-scraping updates a doc instead of duplicating it.

Usage:
  .venv/bin/python db.py import-following [following.json]   # save the following list
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne

import env  # noqa: F401  loads the root .env into os.environ

DEFAULT_DB = "stock_recommender"
DATA = Path(__file__).parent / "data"

_db = None


def get_db():
    global _db
    if _db is None:
        client = MongoClient(os.environ["DB_URL"], tz_aware=True)
        # use the database named in the URL if there is one, else the default
        _db = client.get_default_database(DEFAULT_DB)
        _db.social_accounts.create_index([("platform", ASCENDING), ("user_id", ASCENDING)], unique=True)
        _db.social_posts.create_index([("platform", ASCENDING), ("post_id", ASCENDING)], unique=True)
        _db.social_posts.create_index(
            [("platform", ASCENDING), ("user_id", ASCENDING), ("posted_at", DESCENDING)]
        )
    return _db


def now():
    return datetime.now(timezone.utc)


def account_doc(u, followed_by):
    """Instagram following-list user -> social_accounts doc."""
    return {
        "platform": "instagram",
        "user_id": u["username"],
        "platform_user_id": str(u["pk"]),  # numeric id: stays the same if they rename
        "full_name": u.get("full_name") or None,
        "profile_url": f"https://www.instagram.com/{u['username']}/",
        "profile_pic_url": u.get("profile_pic_url"),
        "is_verified": bool(u.get("is_verified")),
        "is_private": bool(u.get("is_private")),
        "followed_by": str(followed_by),
    }


def save_accounts(users, followed_by):
    ts = now()
    ops = [
        UpdateOne(
            {"platform": "instagram", "user_id": u["username"]},
            {"$set": {**account_doc(u, followed_by), "updated_at": ts}, "$setOnInsert": {"created_at": ts}},
            upsert=True,
        )
        for u in users
    ]
    if not ops:
        return 0, 0
    r = get_db().social_accounts.bulk_write(ops, ordered=False)
    return r.upserted_count, r.modified_count


def post_doc(rec):
    """transcribe_latest.py record -> social_posts doc (no per-segment timestamps)."""
    transcripts = rec.get("transcripts") or []
    posted = rec.get("posted_at")
    return {
        "platform": "instagram",
        "user_id": rec["username"],
        "post_id": rec["code"],  # shortcode from the URL: instagram.com/p/<post_id>/
        "platform_post_id": rec.get("pk"),
        "url": rec["url"],
        "full_name": rec.get("full_name"),
        "media_type": rec.get("media_type"),
        "product_type": rec.get("product_type"),
        "caption": rec.get("caption"),
        "thumbnail_url": rec.get("thumbnail"),
        "posted_at": datetime.fromisoformat(posted) if posted else None,
        "has_audio": bool(transcripts),
        "transcript": "\n\n".join(t["text"] for t in transcripts) or None,
        "language": transcripts[0]["language"] if transcripts else None,
        "duration": round(sum(t.get("duration") or 0 for t in transcripts), 2) if transcripts else None,
        # transcripts from before Maya was added have no "model": they were all Whisper
        "transcript_model": transcripts[0].get("model", "whisper-large-v3-turbo") if transcripts else None,
    }


def save_post(rec):
    ts = now()
    doc = post_doc(rec)
    get_db().social_posts.update_one(
        {"platform": doc["platform"], "post_id": doc["post_id"]},
        {"$set": {**doc, "scraped_at": ts}, "$setOnInsert": {"created_at": ts}},
        upsert=True,
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "import-following":
        sys.exit(__doc__)
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else next(DATA.glob("following_*.json"))
    data = json.loads(path.read_text())
    new, updated = save_accounts(data["users"], data["target_user_id"])
    print(f"social_accounts: {new} new, {updated} updated ({len(data['users'])} in {path.name})")


if __name__ == "__main__":
    main()
