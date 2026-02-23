"""
Local music generation via Meta's AudioCraft (MusicGen).

Install:
  pip install audiocraft          # pulls torch, torchaudio automatically
  pip install soundfile           # for saving WAV output

Models (set AUDIOCRAFT_MODEL in .env):
  facebook/musicgen-small    ~  300 MB  — fast, good for prototyping
  facebook/musicgen-medium   ~  1.5 GB  — better quality
  facebook/musicgen-large    ~  3.3 GB  — best quality
  facebook/musicgen-melody   ~  1.5 GB  — can condition on a melody

MusicGen generates music from a text style prompt. It does NOT sing specific
lyrics — the AUDIOCRAFT_DURATION setting controls how many seconds are produced.
The lyrics are summarised into a thematic description that guides the musical mood.

Config env vars:
  AUDIOCRAFT_MODEL     Model name (default: facebook/musicgen-small)
  AUDIOCRAFT_DEVICE    "cpu" or "cuda" (default: cpu)
  AUDIOCRAFT_DURATION  Seconds of audio to generate (default: 30)
"""

import asyncio
import os
from pathlib import Path
from typing import Optional


class AudioCraftClient:
    """Generates music locally using Meta's MusicGen model."""

    def __init__(self) -> None:
        self.model_name = os.getenv("AUDIOCRAFT_MODEL", "facebook/musicgen-small")
        self.device = os.getenv("AUDIOCRAFT_DEVICE", "cpu")
        self.duration = int(os.getenv("AUDIOCRAFT_DURATION", "30"))
        self._model = None  # lazy-loaded on first call

    # ─────────────────────────────────────────────────────────────
    # Public interface (matches the other music clients)
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
        Generate an instrumental music clip using MusicGen.

        MusicGen works from a text description of style, mood, and instrumentation.
        The lyrics are distilled into a thematic/mood hint rather than being sung
        verbatim — MusicGen produces music, not speech.

        Returns a dict with at minimum:
          audio_url  — absolute local path to the saved WAV file
          provider   — "audiocraft"
          model      — model name used
          duration   — actual duration in seconds
        """
        self._check_imports()

        # Build a concise music-description prompt from the metadata
        parts = [genre, vocal_style]
        if bpm:
            parts.append(f"{bpm} BPM")
        if key:
            parts.append(f"key of {key}")
        style_prompt = ", ".join(p for p in parts if p)

        # Add a lyric-derived mood hint (first ~20 words give MusicGen thematic context)
        if lyrics:
            lyric_preview = " ".join(lyrics.split()[:20])
            prompt = f"{style_prompt}. Mood and theme: {lyric_preview}"
        else:
            prompt = style_prompt

        print(f"[AudioCraft] Model: {self.model_name} | Device: {self.device} | Duration: {self.duration}s")
        print(f"[AudioCraft] Prompt: {prompt[:100]}…" if len(prompt) > 100 else f"[AudioCraft] Prompt: {prompt}")

        # MusicGen is synchronous — run in a thread pool so we don't block the event loop
        loop = asyncio.get_event_loop()
        wav_tensor, sample_rate = await loop.run_in_executor(
            None, self._generate_sync, prompt
        )

        return self._save_audio(wav_tensor, sample_rate, output_dir, filename)

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the MusicGen model (only on first call)."""
        if self._model is None:
            from audiocraft.models import MusicGen
            print(f"[AudioCraft] Loading {self.model_name} on {self.device}… (first run only)")
            self._model = MusicGen.get_pretrained(self.model_name, device=self.device)
            self._model.set_generation_params(duration=self.duration)
            print(f"[AudioCraft] Model loaded.")
        return self._model

    def _generate_sync(self, prompt: str):
        """Run MusicGen synchronously (called inside thread pool executor)."""
        import torch
        model = self._load_model()
        with torch.no_grad():
            wav = model.generate([prompt])   # shape: [batch=1, channels=1, samples]
        # Detach from computation graph, move to CPU, squeeze to 1-D
        wav_np = wav[0, 0].cpu().detach().numpy()
        sample_rate = model.sample_rate
        return wav_np, sample_rate

    def _save_audio(self, wav_np, sample_rate: int, output_dir, filename: str) -> dict:
        """Write the generated waveform to disk as a WAV file."""
        import soundfile as sf
        import numpy as np

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{filename}.wav"
            sf.write(str(out_path), wav_np.astype(np.float32), sample_rate)
            audio_url = str(out_path.resolve())
            print(f"[AudioCraft] Saved → {out_path}")
        else:
            audio_url = "audiocraft://generated_in_memory"
            out_path = None

        actual_duration = len(wav_np) / sample_rate
        return {
            "audio_url": audio_url,
            "duration": round(actual_duration, 2),
            "status": "Success",
            "provider": "audiocraft",
            "model": self.model_name,
            "sample_rate": sample_rate,
        }

    @staticmethod
    def _check_imports() -> None:
        """Raise a clear ImportError if audiocraft or soundfile are not installed."""
        missing = []
        try:
            import audiocraft  # noqa: F401
        except ImportError:
            missing.append("audiocraft")
        try:
            import soundfile  # noqa: F401
        except ImportError:
            missing.append("soundfile")
        if missing:
            raise ImportError(
                f"AudioCraft provider requires: pip install {' '.join(missing)}\n"
                "See: https://github.com/facebookresearch/audiocraft"
            )
