"""
Provider-agnostic music generation with configurable provider chain.

Set MUSIC_PROVIDER (comma-separated) to control which providers are tried and
in what order.  Each provider is attempted in sequence; the first success is
returned.  If all providers fail, RuntimeError is raised.

Supported provider names:
  minimax      — MiniMax music-2.5 API         (requires MINIMAX_API_KEY)
  elevenlabs   — ElevenLabs /v1/music API      (requires ELEVENLABS_API_KEY)
  audiocraft   — Local Meta MusicGen model     (requires: pip install audiocraft soundfile)
  demo         — Simulated response, always succeeds (no API key needed)

Examples:
  MUSIC_PROVIDER=minimax,elevenlabs,audiocraft   # try each in order
  MUSIC_PROVIDER=audiocraft                      # local-only, no API keys
  MUSIC_PROVIDER=demo                            # development / CI

If MUSIC_PROVIDER is not set, the chain is inferred from whichever API keys
are present (minimax → elevenlabs → demo), preserving the original behaviour.
"""

import os
from pathlib import Path
from typing import Optional

from utils.elevenlabs_client import ElevenLabsClient
from utils.minimax_client import MinimaxClient


def _resolve_provider_names() -> list[str]:
    """Return the ordered list of provider names to try."""
    env_value = os.getenv("MUSIC_PROVIDER", "").strip()
    if env_value:
        return [p.strip().lower() for p in env_value.split(",") if p.strip()]

    # Auto-detect from API key presence (legacy behaviour)
    names: list[str] = []
    if os.getenv("MINIMAX_API_KEY"):
        names.append("minimax")
    if os.getenv("ELEVENLABS_API_KEY"):
        names.append("elevenlabs")
    if not names:
        names = ["demo"]
    return names


def _build_provider_chain(
    *,
    lyrics: str,
    vocal_style: str,
    genre: str,
    bpm: Optional[int],
    key: Optional[str],
    output_dir: Optional[Path],
    filename: str,
) -> list[tuple[str, object]]:
    """
    Build the ordered list of ``(display_name, coroutine)`` pairs to try.

    Coroutines are created lazily here; they are only *awaited* inside
    ``generate_music_with_fallback`` when each provider is actually attempted.
    """
    chain: list[tuple[str, object]] = []

    for name in _resolve_provider_names():
        if name == "minimax":
            client = MinimaxClient()
            coro = client.generate_music(
                lyrics=lyrics,
                vocal_style=vocal_style,
                genre=genre,
                bpm=bpm,
                key=key,
            )
            chain.append(("MiniMax", coro))

        elif name == "elevenlabs":
            client = ElevenLabsClient()
            coro = client.generate_music(
                lyrics=lyrics,
                vocal_style=vocal_style,
                genre=genre,
                bpm=bpm,
                key=key,
                output_dir=output_dir,
                filename=filename,
            )
            chain.append(("ElevenLabs", coro))

        elif name == "audiocraft":
            from utils.audiocraft_client import AudioCraftClient  # optional dep
            client = AudioCraftClient()
            coro = client.generate_music(
                lyrics=lyrics,
                vocal_style=vocal_style,
                genre=genre,
                bpm=bpm,
                key=key,
                output_dir=output_dir,
                filename=filename,
            )
            chain.append(("AudioCraft", coro))

        elif name == "demo":
            # Demo uses MinimaxClient in demo mode (no key required)
            client = MinimaxClient()
            coro = client.generate_music(
                lyrics=lyrics,
                vocal_style=vocal_style,
                genre=genre,
                bpm=bpm,
                key=key,
            )
            chain.append(("Demo", coro))

        else:
            print(f"[Music Gen] Unknown provider '{name}' in MUSIC_PROVIDER — skipping.")

    return chain


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
    Attempt music generation with each configured provider in order.

    Returns the first successful result dict (always contains 'audio_url').
    Raises RuntimeError only when every provider has failed.
    """
    chain = _build_provider_chain(
        lyrics=lyrics,
        vocal_style=vocal_style,
        genre=genre,
        bpm=bpm,
        key=key,
        output_dir=output_dir,
        filename=filename,
    )

    if not chain:
        raise RuntimeError(
            "No music providers configured. "
            "Set MUSIC_PROVIDER=minimax|elevenlabs|audiocraft|demo "
            "or add MINIMAX_API_KEY / ELEVENLABS_API_KEY to .env."
        )

    errors: list[str] = []
    for provider_name, coro in chain:
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
