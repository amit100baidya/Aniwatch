"""
user_profile.py
===============
Persistent user profile: watchlist, favourites, and saved anime.
Uses a local JSON file so data survives restarts.
"""

import json
import os
from datetime import datetime

PROFILE_PATH = "user_profile.json"

DEFAULT_PROFILE = {
    "username": "Anime Fan",
    "avatar_color": "#6366f1",
    "joined": "",
    "watchlist":  [],   # list of anime_id dicts
    "favourites": [],
    "saved":      [],
}


def _load() -> dict:
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH) as f:
            return json.load(f)
    profile = DEFAULT_PROFILE.copy()
    profile["joined"] = datetime.now().strftime("%B %Y")
    _save(profile)
    return profile


def _save(profile: dict) -> None:
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)


# ── public API ────────────────────────────────────────────────────────────────

def get_profile() -> dict:
    return _load()


def update_username(name: str) -> None:
    p = _load(); p["username"] = name; _save(p)


def update_avatar_color(color: str) -> None:
    p = _load(); p["avatar_color"] = color; _save(p)


def _entry(anime_id: int, name: str, genres: str, score: float, image_url: str) -> dict:
    return {
        "anime_id": int(anime_id),
        "name": name,
        "genres": genres,
        "score": score,
        "image_url": image_url,
        "added_at": datetime.now().strftime("%d %b %Y"),
    }


def _toggle(list_key: str, anime_id: int, name: str, genres: str, score: float, image_url: str) -> bool:
    """Toggle an item in a list. Returns True if added, False if removed."""
    p = _load()
    ids = [x["anime_id"] for x in p[list_key]]
    if anime_id in ids:
        p[list_key] = [x for x in p[list_key] if x["anime_id"] != anime_id]
        _save(p)
        return False
    else:
        p[list_key].append(_entry(anime_id, name, genres, score, image_url))
        _save(p)
        return True


def toggle_watchlist(anime_id, name, genres, score, image_url) -> bool:
    return _toggle("watchlist", anime_id, name, genres, score, image_url)

def toggle_favourite(anime_id, name, genres, score, image_url) -> bool:
    return _toggle("favourites", anime_id, name, genres, score, image_url)

def toggle_saved(anime_id, name, genres, score, image_url) -> bool:
    return _toggle("saved", anime_id, name, genres, score, image_url)


def is_in(list_key: str, anime_id: int) -> bool:
    p = _load()
    return anime_id in [x["anime_id"] for x in p[list_key]]

def remove_from(list_key: str, anime_id: int) -> None:
    p = _load()
    p[list_key] = [x for x in p[list_key] if x["anime_id"] != anime_id]
    _save(p)

def get_stats() -> dict:
    p = _load()
    return {
        "watchlist":  len(p["watchlist"]),
        "favourites": len(p["favourites"]),
        "saved":      len(p["saved"]),
    }
