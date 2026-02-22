"""
LLMRouter — unified interface for Anthropic (Claude) and Ollama (local) LLMs.

Usage:
    router = LLMRouter.from_env()
    text = await router.complete(system="...", messages=[...])
"""

import json
import os
from typing import Any

import httpx


class LLMRouter:
    """Routes LLM calls to either Anthropic cloud or a local Ollama server."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        ollama_host: str = "http://localhost:11434",
    ):
        self.provider = provider.lower()
        self.ollama_host = ollama_host.rstrip("/")

        if self.provider == "anthropic":
            self.model = model or "claude-opus-4-6"
        elif self.provider == "ollama":
            self.model = model or "deepseek-r1:7b"
        else:
            raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic' or 'ollama'.")

    @classmethod
    def from_env(cls) -> "LLMRouter":
        """Construct from environment variables."""
        return cls(
            provider=os.getenv("FINANCIAL_LLM_PROVIDER", "anthropic"),
            model=os.getenv("FINANCIAL_LLM_MODEL"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 8192,
        stream_label: str = "",
    ) -> str:
        """
        Unified completion interface.

        Args:
            system: System prompt string.
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
            max_tokens: Maximum tokens to generate.
            stream_label: Optional label printed above streamed output.

        Returns:
            Accumulated text response.
        """
        if stream_label:
            print(f"\n{'─' * 60}")
            print(f"  {stream_label}  [{self.provider.upper()} / {self.model}]")
            print(f"{'─' * 60}\n")

        if self.provider == "anthropic":
            return await self._anthropic_complete(system, messages, max_tokens)
        else:
            return await self._ollama_complete(system, messages, max_tokens)

    async def _anthropic_complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        """Stream from Anthropic Claude with adaptive thinking."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        full_text = ""

        async with client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
                full_text += text

        print("\n")
        return full_text

    async def _ollama_complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str:
        """Stream from Ollama via NDJSON /api/chat endpoint."""
        url = f"{self.ollama_host}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }

        full_text = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise RuntimeError(
                        f"Ollama returned HTTP {response.status_code}: {body.decode()[:300]}"
                    )
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        full_text += token
                    if chunk.get("done"):
                        break

        print("\n")
        return full_text

    async def check_ollama_health(self) -> bool:
        """Return True if Ollama is reachable, False otherwise (with a hint)."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(self.ollama_host)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass

        print(
            f"\n[LLMRouter] Cannot reach Ollama at {self.ollama_host}\n"
            "  Start it with:  bash models/start_ollama.sh\n"
            "  Or install via: https://ollama.com/download\n"
        )
        return False

    async def list_local_models(self) -> list[dict]:
        """Return list of pulled Ollama models (empty list if unreachable)."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.ollama_host}/api/tags")
            if r.status_code == 200:
                return r.json().get("models", [])
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        return []
