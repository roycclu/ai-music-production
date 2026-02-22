"""
Async ElevenLabs API client for music/vocal generation.

ElevenLabs docs: https://elevenlabs.io/docs/api-reference
Base URL: https://api.elevenlabs.io

Endpoints used:
  Vocals  : POST /v1/text-to-speech/{voice_id}   → returns audio bytes (MP3)
  Music   : POST /v1/sound-generation             → returns audio bytes (MP3)

Audio bytes are written to disk and a local file path is returned as the
"audio_url" so it integrates cleanly with the rest of the pipeline.

Set ELEVENLABS_API_KEY in .env to enable real generation.
If the key is absent the client runs in DEMO MODE (simulated responses).

Optional env vars:
  ELEVENLABS_VOICE_ID   Voice ID for TTS calls (default: Adam, a warm male voice)
  ELEVENLABS_BASE_URL   Override if needed
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import httpx

ELEVENLABS_BASE_URL = os.getenv(
    "ELEVENLABS_BASE_URL", "https://api.elevenlabs.io"
)

# A stable built-in ElevenLabs voice that suits a warm folk/soul delivery.
# Users can override via ELEVENLABS_VOICE_ID in .env.
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Adam


class ElevenLabsClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)

        # Demo mode when no key is configured
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
            "Accept": "audio/mpeg",
        }

    # ─────────────────────────────────────────────────────────────
    # Music / vocal generation
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
        Generate a vocal/music track via ElevenLabs TTS.

        For vocal tracks (Agent 2) this produces a human-sounding reading of
        the lyrics using a configurable voice — a degraded-but-functional fallback
        when MiniMax is unavailable.

        Saves audio bytes to output_dir/filename.mp3 and returns a dict with:
          audio_url  — absolute path to the saved file (file:///...)
          provider   — "elevenlabs"
          duration   — estimated duration in seconds
        """
        if self.demo_mode:
            print("[ElevenLabs DEMO] Simulating music generation (1 s delay)…")
            await asyncio.sleep(1)
            return {
                "audio_url": f"https://demo.elevenlabs.example/audio/{filename}.mp3",
                "duration": 240.0,
                "task_id": f"demo_elevenlabs_{abs(hash(lyrics)) % 10_000:04d}",
                "status": "Success",
                "provider": "elevenlabs_demo",
            }

        # Build a descriptive text prompt that merges song metadata with lyrics.
        # ElevenLabs TTS will read this as the vocal track.
        bpm_note = f" at {bpm} BPM" if bpm else ""
        key_note = f" in {key}" if key else ""
        tts_text = (
            f"[{genre.title()} song{key_note}{bpm_note} — {vocal_style}]\n\n{lyrics}"
        )

        try:
            audio_bytes = await self._tts(tts_text)
        except Exception:
            # Secondary attempt: sound-generation endpoint for ambient music content
            description = (
                f"{genre} music{key_note}{bpm_note}. "
                f"Vocal style: {vocal_style}. "
                f"Opening lyrics: {lyrics[:300]}"
            )
            audio_bytes = await self._sound_generation(description, duration_seconds=22.0)

        return self._save_audio(audio_bytes, output_dir, filename)

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    async def _tts(self, text: str) -> bytes:
        """Call the ElevenLabs text-to-speech endpoint; return raw audio bytes."""
        url = f"{ELEVENLABS_BASE_URL}/v1/text-to-speech/{self.voice_id}"
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.75,
                "style": 0.35,
                "use_speaker_boost": True,
            },
        }
        async with httpx.AsyncClient(timeout=120.0) as http:
            response = await http.post(url, headers=self._headers, json=payload)
            response.raise_for_status()
            return response.content

    async def _sound_generation(self, description: str, duration_seconds: float = 22.0) -> bytes:
        """
        Call the ElevenLabs sound-generation endpoint; return raw audio bytes.
        Duration is capped at 22 s (API limit).
        """
        url = f"{ELEVENLABS_BASE_URL}/v1/sound-generation"
        payload = {
            "text": description,
            "duration_seconds": min(duration_seconds, 22.0),
            "prompt_influence": 0.35,
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

        # Rough duration estimate: ~1 minute per ~5 000 bytes for MP3 at 128 kbps
        est_duration = max(10.0, len(audio_bytes) / 5_000 * 60)

        return {
            "audio_url": audio_url,
            "duration": round(est_duration, 1),
            "status": "Success",
            "provider": "elevenlabs",
        }
