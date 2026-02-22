"""
File I/O utilities, output directory management, and per-stage cache helpers.
"""

import json
from pathlib import Path

# ── Output directories (relative to project root) ─────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MELODIES_DIR = BASE_DIR / "Melodies"
VOCALS_DIR = BASE_DIR / "Vocals"
SONGS_DIR = BASE_DIR / "Songs"
MUSIC_VIDEO_DIR = BASE_DIR / "Music Video"


def create_output_dirs() -> None:
    """Create all agent output directories if they don't already exist."""
    for directory in [MELODIES_DIR, VOCALS_DIR, SONGS_DIR, MUSIC_VIDEO_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[Setup] {directory.relative_to(BASE_DIR)}/")


def save_json(path: Path, data: dict | list, indent: int = 2) -> None:
    """Write data as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)
    print(f"[Saved] {path.relative_to(BASE_DIR)}")


def save_text(path: Path, content: str) -> None:
    """Write a plain-text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"[Saved] {path.relative_to(BASE_DIR)}")


def load_json(path: Path) -> dict:
    """Load JSON from a file. Returns empty dict if file is missing."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def load_text(path: Path) -> str:
    """Load plain text from a file. Returns empty string if file is missing."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


# ── Per-stage cache checks ─────────────────────────────────────────────────────

def melody_cache_valid() -> bool:
    """True when the songwriter's primary output file exists."""
    return (MELODIES_DIR / "song_structure.json").exists()


def vocals_cache_valid() -> bool:
    """True when both the vocal direction JSON and audio URL file exist."""
    return (
        (VOCALS_DIR / "vocal_direction.json").exists()
        and (VOCALS_DIR / "audio_url.txt").exists()
    )


def production_cache_valid() -> bool:
    """True when both the production brief JSON and audio URL file exist."""
    return (
        (SONGS_DIR / "production_brief.json").exists()
        and (SONGS_DIR / "audio_url.txt").exists()
    )


def video_cache_valid() -> bool:
    """True when the video brief JSON and URL manifest both exist."""
    return (
        (MUSIC_VIDEO_DIR / "video_brief.json").exists()
        and (MUSIC_VIDEO_DIR / "final_video_urls.json").exists()
    )


# ── Per-stage cache loaders ────────────────────────────────────────────────────

def load_melody_cache() -> dict:
    """Reconstruct the songwriter agent's return dict from persisted files."""
    return load_json(MELODIES_DIR / "song_structure.json")


def load_vocals_cache() -> dict:
    """Reconstruct the singer agent's return dict from persisted files."""
    vocal_data = load_json(VOCALS_DIR / "vocal_direction.json")
    audio_url = load_text(VOCALS_DIR / "audio_url.txt") or "N/A"
    audio_result = load_json(VOCALS_DIR / "audio_result.json")
    return {**vocal_data, "audio_url": audio_url, "audio_result": audio_result}


def load_production_cache() -> dict:
    """Reconstruct the producer agent's return dict from persisted files."""
    production_data = load_json(SONGS_DIR / "production_brief.json")
    audio_url = load_text(SONGS_DIR / "audio_url.txt") or "N/A"
    audio_result = load_json(SONGS_DIR / "audio_result.json")
    return {**production_data, "audio_url": audio_url, "audio_result": audio_result}


def load_video_cache() -> dict:
    """Reconstruct the video producer agent's return dict from persisted files."""
    video_brief = load_json(MUSIC_VIDEO_DIR / "video_brief.json")
    # Load individual scene files (scene_01_*.json …) sorted by name
    scene_files = sorted(MUSIC_VIDEO_DIR.glob("scene_[0-9][0-9]_*.json"))
    scene_results = [load_json(f) for f in scene_files]
    return {
        **video_brief,
        "scenes": scene_results,
        "video_urls": [s.get("video_url", "N/A") for s in scene_results],
    }


def format_lyrics_with_markers(song_data: dict) -> str:
    """
    Return the full lyrics formatted with [Section] markers as expected by
    MiniMax music-01 and by human readers.
    """
    lyrics = song_data.get("lyrics", {})
    section_map = [
        ("verse_1",    "[Verse 1]"),
        ("pre_chorus", "[Pre-Chorus]"),
        ("chorus",     "[Chorus]"),
        ("verse_2",    "[Verse 2]"),
        ("pre_chorus", "[Pre-Chorus]"),
        ("chorus",     "[Chorus]"),
        ("bridge",     "[Bridge]"),
        ("outro",      "[Outro]"),
    ]
    parts = []
    for key, label in section_map:
        text = lyrics.get(key, "").strip()
        if text:
            parts.append(f"{label}\n{text}")
    return "\n\n".join(parts)
