"""
Async ElevenLabs API client for music generation.

ElevenLabs docs: https://elevenlabs.io/docs/api-reference
Base URL: https://api.elevenlabs.io

Endpoints used:
  Music   : POST /v1/music              → full song generation (prompt + lyrics)
  Sound   : POST /v1/sound-generation   → short ambient/instrumental (≤ 22 s, fallback)

Audio bytes are written to disk and a local file path is returned as the
"audio_url" so it integrates cleanly with the rest of the pipeline.

Set ELEVENLABS_API_KEY in .env to enable real generation.
If the key is absent the client runs in DEMO MODE (simulated responses).

Optional env vars:
  ELEVENLABS_BASE_URL   Override API base (default: https://api.elevenlabs.io)
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import httpx

ELEVENLABS_BASE_URL = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")


class ElevenLabsClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")

        explicit_demo = os.getenv("DEMO_MODE", "false").lower() == "true"
        self.demo_mode = explicit_demo or not self.api_key

        if self.demo_mode and not self.api_key:
            print(
                "\n[ElevenLabs] No ELEVENLABS_API_KEY found — "
                "running in DEMO MODE (simulated responses, no real API calls)\n"
            )

        self._headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    # ─────────────────────────────────────────────────────────────
    # Music generation
    # ─────────────────────────────────────────────────────────────

    async def generate_music(
        self,
        lyrics: str,
        vocal_style: str,
        genre: str = "folk",
        bpm: Optional[int] = None,
        key: Optional[str] = None,
        output_dir: Optional[Path] = None,
        filename: str = "audio_track",
    ) -> dict:
        """
        Generate a music track via ElevenLabs /v1/music (song generation).

        Builds a style prompt from the genre/vocal_style/bpm/key metadata and
        passes the full lyrics to the music generation endpoint so the output is
        an actual song, not a text-to-speech voiceover.

        Falls back to /v1/sound-generation (instrumental, ≤ 22 s) if the music
        endpoint is unavailable or returns an error.

        Returns a dict with at minimum:
          audio_url  — absolute local file path (file saved to output_dir)
          provider   — "elevenlabs"
        """
        if self.demo_mode:
            print("[ElevenLabs DEMO] Simulating music generation (1 s delay)…")
            await asyncio.sleep(1)
            return {
                "audio_url": f"https://demo.elevenlabs.example/audio/{filename}.mp3",
                "duration": 180.0,
                "task_id": f"demo_elevenlabs_{abs(hash(lyrics)) % 10_000:04d}",
                "status": "Success",
                "provider": "elevenlabs_demo",
            }

        # Build a concise music style prompt — NOT the lyrics text itself
        style_parts = [genre, vocal_style]
        if bpm:
            style_parts.append(f"{bpm} BPM")
        if key:
            style_parts.append(f"key of {key}")
        style_prompt = ", ".join(p for p in style_parts if p)

        # Primary: music generation endpoint (full song with lyrics)
        try:
            audio_bytes = await self._music_generation(style_prompt, lyrics)
            print("[ElevenLabs] Music generation succeeded via /v1/music")
        except Exception as exc:
            print(f"[ElevenLabs] /v1/music failed ({exc}), trying sound-generation…")
            # Fallback: short instrumental clip from style description (no lyrics, ≤ 22 s)
            audio_bytes = await self._sound_generation(style_prompt, duration_seconds=22.0)
            print("[ElevenLabs] Sound generation fallback succeeded")

        return self._save_audio(audio_bytes, output_dir, filename)

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    async def _music_generation(self, prompt: str, lyrics: str) -> bytes:
        """
        POST /v1/music — ElevenLabs full song generation.

        Accepts a style prompt and optional lyrics to produce a complete
        song with melody, harmony, and (optionally) vocals.
        """
        url = f"{ELEVENLABS_BASE_URL}/v1/music"
        payload: dict = {"prompt": prompt}
        if lyrics:
            payload["lyrics"] = lyrics
        headers = {**self._headers, "Accept": "audio/mpeg"}
        async with httpx.AsyncClient(timeout=180.0) as http:
            response = await http.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.content

    async def _sound_generation(self, description: str, duration_seconds: float = 22.0) -> bytes:
        """
        POST /v1/sound-generation — short ambient/instrumental clip.

        Duration is capped at 22 s (API limit). Used as internal fallback when
        the /v1/music endpoint is unavailable.
        """
        url = f"{ELEVENLABS_BASE_URL}/v1/sound-generation"
        payload = {
            "text": description,
            "duration_seconds": min(duration_seconds, 22.0),
            "prompt_influence": 0.5,
        }
        headers = {**self._headers, "Accept": "audio/mpeg"}
        async with httpx.AsyncClient(timeout=120.0) as http:
            response = await http.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.content

    @staticmethod
    def _save_audio(
        audio_bytes: bytes,
        output_dir: Optional[Path],
        filename: str,
    ) -> dict:
        """Write audio bytes to disk and return a pipeline-compatible result dict."""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{filename}.mp3"
            out_path.write_bytes(audio_bytes)
            audio_url = str(out_path.resolve())
            print(f"[ElevenLabs] Saved audio → {out_path}")
        else:
            audio_url = "elevenlabs://generated_in_memory"

        est_duration = max(10.0, len(audio_bytes) / 5_000 * 60)
        return {
            "audio_url": audio_url,
            "duration": round(est_duration, 1),
            "status": "Success",
            "provider": "elevenlabs",
        }
