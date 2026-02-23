#!/usr/bin/env python3
"""
Multi-Agent Music Video Production Pipeline
============================================

Orchestrates four specialized AI agents — Songwriter, Singer, Music Producer,
and Music Video Producer — to create a complete original music video from scratch.

Pipeline:
  Agent 1 (Songwriter)       → Melodies/
  Agent 2 (Singer)           → Vocals/
  Agent 3 (Music Producer)   → Songs/
  Agent 4 (Video Producer)   → Music Video/
  Final deliverable          → final_deliverable.json

Usage:
  python main.py

Prerequisites:
  cp .env.example .env   # fill in API keys
  pip install -r requirements.txt
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import anthropic

from agents.songwriter import run_songwriter
from agents.singer import run_singer
from agents.producer import run_producer
from agents.video_producer import run_video_producer
from utils.observability import setup_observability
from utils.file_utils import (
    BASE_DIR,
    MELODIES_DIR,
    MUSIC_VIDEO_DIR,
    SONGS_DIR,
    VOCALS_DIR,
    create_output_dirs,
    load_melody_cache,
    load_vocals_cache,
    load_production_cache,
    load_video_cache,
    melody_cache_valid,
    vocals_cache_valid,
    production_cache_valid,
    video_cache_valid,
    save_json,
)

load_dotenv()


# ── Orchestrator ───────────────────────────────────────────────────────────────

async def main(
    force: bool = False,
    force_song: bool = False,
    force_video: bool = False,
    interactive: bool = True,
) -> dict:
    """
    Main pipeline orchestrator.

    Runs all four agents sequentially, passing each agent's output as
    context to the next. Each stage is skipped if its output files already
    exist on disk (cache hit).

    Args:
        force:        Regenerate all four stages regardless of cache.
        force_song:   Regenerate agents 1–3 (songwriter, singer, producer).
        force_video:  Regenerate agent 4 (video producer).
        interactive:  When True (default), prompt the user before each cached
                      stage so they can choose to regenerate selectively.
                      Ignored when force=True.

    Returns the compiled final deliverable package.
    """
    _print_banner(force)
    _check_env()
    create_output_dirs()
    setup_observability()

    # ── Interactive selective-regeneration prompts ─────────────────────────────
    if not force and interactive:
        _force_song, _force_video = _prompt_regeneration()
        force_song  = force_song  or _force_song
        force_video = force_video or _force_video

    redo_song  = force or force_song
    redo_video = force or force_video

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── Agent 1 — Songwriter ───────────────────────────────────────────────────
    if not redo_song and melody_cache_valid():
        song_data = load_melody_cache()
        _print_cache_hit("Agent 1 — Songwriter", "Melodies/song_structure.json")
    else:
        song_data = await _run_agent(
            label="Agent 1 — Songwriter",
            coro=run_songwriter(client),
            summary_fn=lambda d: (
                f"  Song:  '{d.get('title')}'\n"
                f"  Key:   {d.get('key')}  |  BPM: {d.get('bpm')}  "
                f"|  Time: {d.get('time_signature')}\n"
                f"  Files: Melodies/"
            ),
        )

    # ── Agent 2 — Singer ──────────────────────────────────────────────────────
    if not redo_song and vocals_cache_valid():
        vocal_data = load_vocals_cache()
        _print_cache_hit("Agent 2 — Singer", "Vocals/vocal_direction.json + audio_url.txt")
    else:
        vocal_data = await _run_agent(
            label="Agent 2 — Singer",
            coro=run_singer(client, song_data),
            summary_fn=lambda d: (
                f"  Style: {str(d.get('vocal_direction', {}).get('style', ''))[:80]}…\n"
                f"  Audio: {d.get('audio_url', 'N/A')}\n"
                f"  Files: Vocals/"
            ),
        )

    # ── Agent 3 — Music Producer ───────────────────────────────────────────────
    if not redo_song and production_cache_valid():
        production_data = load_production_cache()
        _print_cache_hit("Agent 3 — Music Producer", "Songs/production_brief.json + audio_url.txt")
    else:
        production_data = await _run_agent(
            label="Agent 3 — Music Producer",
            coro=run_producer(client, song_data, vocal_data),
            summary_fn=lambda d: (
                f"  Style: {str(d.get('production_brief', {}).get('style', ''))[:80]}…\n"
                f"  Track: {d.get('audio_url', 'N/A')}\n"
                f"  Files: Songs/"
            ),
        )

    # ── Agent 4 — Music Video Producer ────────────────────────────────────────
    if not redo_video and video_cache_valid():
        video_data = load_video_cache()
        _print_cache_hit("Agent 4 — Music Video Producer", "Music Video/video_brief.json + final_video_urls.json")
    else:
        video_data = await _run_agent(
            label="Agent 4 — Music Video Producer",
            coro=run_video_producer(client, song_data, vocal_data, production_data),
            summary_fn=lambda d: (
                f"  Scenes: {len(d.get('scenes', []))}\n"
                f"  Files:  Music Video/"
            ),
        )

    # ── Compile + save final package ──────────────────────────────────────────
    package = _compile_package(song_data, vocal_data, production_data, video_data)
    save_json(BASE_DIR / "final_deliverable.json", package)

    _print_final_summary(package)
    return package


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _run_agent(label: str, coro, summary_fn) -> dict:
    """Run an agent coroutine with error handling and a summary print."""
    try:
        result = await coro
    except Exception as exc:
        print(f"\n[ERROR] {label} failed: {exc}", file=sys.stderr)
        raise

    print(f"\n{'─' * 60}")
    print(f"  ✓ {label} complete")
    print(summary_fn(result))
    print("─" * 60)
    return result


def _compile_package(
    song_data: dict,
    vocal_data: dict,
    production_data: dict,
    video_data: dict,
) -> dict:
    """Assemble the final deliverable from all agent outputs."""
    return {
        "production_package": {
            "title":      song_data.get("title"),
            "created_at": datetime.now().isoformat(),
        },
        "song": {
            "title":          song_data.get("title"),
            "key":            song_data.get("key"),
            "bpm":            song_data.get("bpm"),
            "time_signature": song_data.get("time_signature"),
            "full_lyrics":    song_data.get("lyrics", {}),
            "chord_chart":    song_data.get("chord_chart", ""),
        },
        "vocals": {
            "style":     vocal_data.get("vocal_direction", {}).get("style"),
            "audio_url": vocal_data.get("audio_url"),
        },
        "production": {
            "style":           production_data.get("production_brief", {}).get("style"),
            "instrumentation": production_data.get("production_brief", {}).get("instrumentation", []),
            "final_track_url": production_data.get("audio_url"),
        },
        "video": {
            "treatment":    video_data.get("treatment"),
            "total_scenes": len(video_data.get("scenes", [])),
            "timeline": [
                {
                    "scene":    s.get("scene_number"),
                    "section":  s.get("section_label"),
                    "duration": s.get("duration_seconds"),
                    "url":      s.get("video_url"),
                }
                for s in video_data.get("scenes", [])
            ],
        },
        "output_directories": {
            "Melodies":    str(MELODIES_DIR),
            "Vocals":      str(VOCALS_DIR),
            "Songs":       str(SONGS_DIR),
            "Music Video": str(MUSIC_VIDEO_DIR),
        },
    }


def _prompt_regeneration() -> tuple[bool, bool]:
    """
    Ask the user whether to regenerate the song and/or music video.

    Only prompts for stages that already have cached output — there is no
    point asking "redo?" when nothing has been produced yet.

    Returns:
        (force_song, force_video) — True means regenerate that group.
    """
    song_cached  = melody_cache_valid() or vocals_cache_valid() or production_cache_valid()
    video_cached = video_cache_valid()

    if not song_cached and not video_cached:
        return False, False  # nothing cached — just generate everything

    print("─" * 60)
    print("  PRODUCTION OPTIONS")
    print("─" * 60)

    force_song = False
    if song_cached:
        cached = []
        if melody_cache_valid():
            cached.append("Melodies")
        if vocals_cache_valid():
            cached.append("Vocals")
        if production_cache_valid():
            cached.append("Songs")
        print(f"  Song cached:   {' + '.join(cached)}/")
        ans = input("  Reproduce song (agents 1-3)?      [y/N] ").strip().lower()
        force_song = ans in ("y", "yes")

    force_video = False
    if video_cached:
        print("  Video cached:  Music Video/")
        ans = input("  Reproduce music video (agent 4)?  [y/N] ").strip().lower()
        force_video = ans in ("y", "yes")

    print("─" * 60 + "\n")
    return force_song, force_video


def _print_cache_hit(label: str, files: str) -> None:
    """Print a short notice that an agent stage was skipped via cache."""
    print(f"\n{'─' * 60}")
    print(f"  ↩  {label} — loaded from cache")
    print(f"     {files}")
    print("─" * 60)


def _print_banner(force: bool = False) -> None:
    demo = os.getenv("DEMO_MODE", "false").upper()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    has_minimax = bool(os.getenv("MINIMAX_API_KEY"))
    has_elevenlabs = bool(os.getenv("ELEVENLABS_API_KEY"))

    if has_minimax and has_elevenlabs:
        music_providers = "MiniMax (primary) → ElevenLabs (fallback)"
    elif has_minimax:
        music_providers = "MiniMax only (add ELEVENLABS_API_KEY for fallback)"
    elif has_elevenlabs:
        music_providers = "ElevenLabs only (MINIMAX_API_KEY not set)"
    else:
        music_providers = "demo mode (no API keys configured)"

    print("\n" + "═" * 60)
    print("  MULTI-AGENT MUSIC VIDEO PRODUCTION PIPELINE")
    print("═" * 60)
    print(f"  Started:   {ts}")
    print(f"  Model:     claude-opus-4-6 (adaptive thinking)")
    print(f"  Music:     {music_providers}")
    print(f"  Demo mode: {demo}")
    print(f"  Cache:     {'DISABLED (--force)' if force else 'enabled — skips completed stages'}")
    print("═" * 60 + "\n")


def _check_env() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit(
            "[ERROR] ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example → .env and add your key."
        )
    has_minimax = bool(os.getenv("MINIMAX_API_KEY"))
    has_elevenlabs = bool(os.getenv("ELEVENLABS_API_KEY"))
    if not has_minimax and not has_elevenlabs:
        print(
            "[WARNING] Neither MINIMAX_API_KEY nor ELEVENLABS_API_KEY is set — "
            "music generation will run in demo mode (no real audio generated).\n"
        )
    elif not has_minimax:
        print(
            "[INFO] MINIMAX_API_KEY not set — music generation will use "
            "ElevenLabs only (no MiniMax fallback).\n"
        )
    elif not has_elevenlabs:
        print(
            "[INFO] ELEVENLABS_API_KEY not set — if MiniMax fails, the pipeline "
            "will stop (no ElevenLabs fallback available).\n"
        )


def _print_final_summary(package: dict) -> None:
    song = package.get("song", {})
    vocals = package.get("vocals", {})
    production = package.get("production", {})
    video = package.get("video", {})

    print("\n" + "═" * 60)
    print("  FINAL PRODUCTION PACKAGE")
    print("═" * 60)
    print(f"\n  Song:  '{song.get('title')}'")
    print(f"  Key:   {song.get('key')}  |  BPM: {song.get('bpm')}  |  {song.get('time_signature')}")

    print(f"\n  Vocal track URL:")
    print(f"    {vocals.get('audio_url', 'N/A')}")

    print(f"\n  Final music track URL:")
    print(f"    {production.get('final_track_url', 'N/A')}")

    timeline = video.get("timeline", [])
    print(f"\n  Music video ({video.get('total_scenes', 0)} scenes):")
    for entry in timeline:
        print(
            f"    Scene {entry.get('scene'):>2}  "
            f"[{entry.get('section', ''):<15}]  "
            f"{entry.get('duration', 0):>3}s  "
            f"{entry.get('url', 'N/A')}"
        )

    dirs = package.get("output_directories", {})
    print(f"\n  Output directories:")
    for name, path in dirs.items():
        print(f"    {name:<12} → {path}")

    print(f"\n  Full package: {BASE_DIR / 'final_deliverable.json'}")
    print("═" * 60 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Music Video Production Pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached outputs and regenerate every stage from scratch",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip prompts and use cached outputs without asking (useful for scripts/CI)",
    )
    args = parser.parse_args()
    asyncio.run(main(force=args.force, interactive=not args.no_interactive))
