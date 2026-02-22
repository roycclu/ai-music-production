"""
Provider-agnostic music generation with automatic fallback.

Priority order:
  1. MiniMax music-01  (primary — requires MINIMAX_API_KEY)
  2. ElevenLabs        (fallback — requires ELEVENLABS_API_KEY)
  3. Demo mode         (last resort — when no real API keys are configured)

The pipeline stops only if every real provider raises an exception.
Demo mode is used as a final safety net so development runs always complete.
"""

import os
from pathlib import Path
from typing import Optional

from utils.elevenlabs_client import ElevenLabsClient
from utils.minimax_client import MinimaxClient


async def generate_music_with_fallback(
    *,
    lyrics: str,
    vocal_style: str,
    genre: str = "folk",
    bpm: Optional[int] = None,
    key: Optional[str] = None,
    output_dir: Optional[Path] = None,
    filename: str = "audio_track",
) -> dict:
    """
    Attempt music generation with each configured provider in priority order.

    Returns the first successful result dict (always contains 'audio_url').
    Raises RuntimeError only when every real provider has failed.

    Provider selection logic:
    - If MINIMAX_API_KEY is set     → MiniMax is tried first
    - If ELEVENLABS_API_KEY is set  → ElevenLabs is tried next (or first if no MiniMax key)
    - If neither key is set         → MiniMax demo mode is used (always succeeds)
    """
    minimax = MinimaxClient()
    elevenlabs = ElevenLabsClient()

    # Build ordered provider list — only real (non-demo) providers are included
    # so a demo success from one doesn't mask a real API that could succeed.
    providers: list[tuple[str, object]] = []

    if os.getenv("MINIMAX_API_KEY"):
        providers.append((
            "MiniMax",
            minimax.generate_music(
                lyrics=lyrics,
                vocal_style=vocal_style,
                genre=genre,
                bpm=bpm,
                key=key,
            ),
        ))

    if os.getenv("ELEVENLABS_API_KEY"):
        providers.append((
            "ElevenLabs",
            elevenlabs.generate_music(
                lyrics=lyrics,
                vocal_style=vocal_style,
                genre=genre,
                bpm=bpm,
                key=key,
                output_dir=output_dir,
                filename=filename,
            ),
        ))

    # No real keys configured — fall back to MiniMax demo mode
    if not providers:
        print("[Music Gen] No API keys configured — using MiniMax demo mode.")
        result = await minimax.generate_music(
            lyrics=lyrics,
            vocal_style=vocal_style,
            genre=genre,
            bpm=bpm,
            key=key,
        )
        result.setdefault("provider", "MiniMax (demo)")
        return result

    # Try each real provider in order; collect errors
    errors: list[str] = []
    for provider_name, coro in providers:
        try:
            print(f"\n[Music Gen] Trying {provider_name}…")
            result = await coro
            result.setdefault("provider", provider_name)
            print(f"[Music Gen] {provider_name} succeeded.")
            return result
        except Exception as exc:
            print(f"[Music Gen] {provider_name} failed: {exc}")
            errors.append(f"{provider_name}: {exc}")

    raise RuntimeError(
        "All music generation providers failed:\n" + "\n".join(errors)
    )
