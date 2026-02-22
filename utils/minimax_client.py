"""
Async MiniMax API client for music (music-01) and video (video-01) generation.

MiniMax API docs: https://www.minimax.chat/
Base URL: https://api.minimax.chat/v1

Audio endpoint : POST /music_generation
Video endpoint : POST /video_generation
Video poll     : GET  /query/video_generation?task_id=...

If your MiniMax subscription uses different endpoints, update MINIMAX_BASE_URL
and the endpoint paths in _post() below.
"""

import asyncio
import os
import time
from typing import Optional

import httpx

MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")


class MinimaxClient:
    def __init__(self):
        self.api_key = os.getenv("MINIMAX_API_KEY", "")
        self.group_id = os.getenv("MINIMAX_GROUP_ID", "")

        # Auto-enable demo mode when no API key is configured
        explicit_demo = os.getenv("DEMO_MODE", "false").lower() == "true"
        self.demo_mode = explicit_demo or not self.api_key

        if self.demo_mode:
            if not self.api_key:
                print(
                    "\n[MiniMax] No MINIMAX_API_KEY found — running in DEMO MODE "
                    "(simulated responses, no real API calls)\n"
                )
            else:
                print("\n[MiniMax] DEMO_MODE=true — using simulated responses\n")

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.group_id:
            # Some MiniMax plans require the Group ID as a header
            self._headers["GroupId"] = self.group_id

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
        retry: bool = True,
    ) -> dict:
        """
        Generate music via MiniMax music-01.

        Returns a dict containing at minimum:
          audio_url  — direct download URL or presigned URL
          duration   — track duration in seconds
        """
        if self.demo_mode:
            print("[MiniMax DEMO] Simulating music generation (2 s delay)...")
            await asyncio.sleep(2)
            return {
                "audio_url": "https://demo.minimax.example/music/track_demo.mp3",
                "duration": 240.0,
                "task_id": "demo_music_task_001",
                "status": "Success",
            }

        # ── Build payload ──────────────────────────────────────────
        # Adjust field names to match your MiniMax API version.
        payload: dict = {
            "model": "music-01",
            "lyrics": lyrics,
            "vocal_style": vocal_style,
            "genre": genre,
        }
        if bpm:
            payload["bpm"] = bpm
        if key:
            payload["key"] = key

        try:
            result = await self._post("/music_generation", payload)

            # Handle async task pattern (task_id returned instead of audio URL)
            if "task_id" in result and "audio_url" not in result:
                return await self._poll_music_task(result["task_id"])

            return result

        except Exception as exc:
            if retry:
                print(f"[MiniMax] Music generation failed ({exc}), retrying once...")
                await asyncio.sleep(5)
                return await self.generate_music(
                    lyrics, vocal_style, genre, bpm, key, retry=False
                )
            raise

    # ─────────────────────────────────────────────────────────────
    # Video generation
    # ─────────────────────────────────────────────────────────────

    async def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        retry: bool = True,
    ) -> dict:
        """
        Generate a video clip via MiniMax video-01.

        Video generation is always asynchronous; this method polls until
        the task completes and returns a dict with at minimum:
          video_url  — direct download or streaming URL
          duration   — clip duration in seconds
        """
        if self.demo_mode:
            # Unique-ish URL per scene so outputs are distinguishable
            scene_id = abs(hash(prompt)) % 10_000
            print(
                f"[MiniMax DEMO] Simulating video generation "
                f"(scene id {scene_id:04d}, 1 s delay)..."
            )
            await asyncio.sleep(1)
            return {
                "video_url": (
                    f"https://demo.minimax.example/video/scene_{scene_id:04d}.mp4"
                ),
                "duration": duration,
                "task_id": f"demo_video_task_{scene_id:04d}",
                "status": "Success",
            }

        payload: dict = {
            "model": "video-01",
            "prompt": prompt,
            "duration": duration,
        }

        try:
            result = await self._post("/video_generation", payload)

            # Video generation is always async in MiniMax
            task_id = result.get("task_id")
            if task_id:
                return await self._poll_video_task(task_id)

            return result

        except Exception as exc:
            if retry:
                refined = f"{prompt}, cinematic quality, film grain"
                print(
                    f"[MiniMax] Video generation failed ({exc}), "
                    f"retrying with refined prompt..."
                )
                await asyncio.sleep(5)
                return await self.generate_video(refined, duration, retry=False)
            raise

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    async def _post(self, endpoint: str, payload: dict) -> dict:
        """POST to MiniMax API and return parsed JSON."""
        url = f"{MINIMAX_BASE_URL}{endpoint}"
        async with httpx.AsyncClient(timeout=60.0) as http:
            response = await http.post(url, headers=self._headers, json=payload)
            response.raise_for_status()
            data = response.json()

        # MiniMax wraps errors in base_resp
        if "base_resp" in data:
            base = data["base_resp"]
            if base.get("status_code", 0) != 0:
                raise RuntimeError(
                    f"MiniMax error {base.get('status_code')}: "
                    f"{base.get('status_msg', 'unknown error')}"
                )

        return data

    async def _poll_music_task(self, task_id: str, timeout: int = 180) -> dict:
        """Poll until the music generation task finishes."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(5)
            async with httpx.AsyncClient(timeout=30.0) as http:
                response = await http.get(
                    f"{MINIMAX_BASE_URL}/query/music_generation",
                    headers=self._headers,
                    params={"task_id": task_id},
                )
                response.raise_for_status()
                result = response.json()

            status = (
                result.get("status")
                or result.get("data", {}).get("status", "")
            ).lower()

            if status in ("success", "completed"):
                return result.get("data", result)
            if status in ("failed", "error"):
                raise RuntimeError(f"MiniMax music task failed: {result}")

            print(f"  [MiniMax] Music task {task_id!r} → {status}")

        raise TimeoutError(f"Music generation timed out after {timeout}s")

    async def _poll_video_task(self, task_id: str, timeout: int = 360) -> dict:
        """Poll until the video generation task finishes."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(10)
            async with httpx.AsyncClient(timeout=30.0) as http:
                response = await http.get(
                    f"{MINIMAX_BASE_URL}/query/video_generation",
                    headers=self._headers,
                    params={"task_id": task_id},
                )
                response.raise_for_status()
                result = response.json()

            status = result.get("status", "").lower()

            if status in ("success", "completed"):
                return result
            if status in ("failed", "error"):
                raise RuntimeError(f"MiniMax video task failed: {result}")

            print(f"  [MiniMax] Video task {task_id!r} → {status}")

        raise TimeoutError(f"Video generation timed out after {timeout}s")
