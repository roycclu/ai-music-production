"""
Shared Claude API helpers used by all agents.
"""

import json
import re
from typing import Any

import anthropic


async def call_claude(
    client: anthropic.AsyncAnthropic,
    system: str,
    user_content: str,
    max_tokens: int = 8192,
    label: str = "",
) -> str:
    """
    Call Claude claude-opus-4-6 with adaptive thinking and streaming.
    Prints output in real-time and returns the accumulated text response.
    """
    if label:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}\n")

    full_text = ""

    async with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        # text_stream filters out thinking blocks automatically
        async for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text

    print("\n")
    return full_text


def extract_json(text: str) -> Any:
    """
    Robustly extract JSON from Claude's response text.

    Tries (in order):
      1. ```json ... ``` fenced blocks (last occurrence wins)
      2. Any ``` ... ``` fenced block containing a dict/list
      3. Balanced-brace scanning for a raw JSON object
    """
    # 1. Try ```json ... ``` blocks
    json_blocks = re.findall(r"```json\s*([\s\S]*?)\s*```", text)
    for block in reversed(json_blocks):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # 2. Try any fenced code block
    code_blocks = re.findall(r"```\s*([\s\S]*?)\s*```", text)
    for block in reversed(code_blocks):
        stripped = block.strip()
        if not stripped.startswith(("{", "[")):
            continue
        try:
            result = json.loads(stripped)
            if isinstance(result, (dict, list)):
                return result
        except json.JSONDecodeError:
            continue

    # 3. Balanced-brace scan — finds the first top-level JSON object
    brace_depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0 and start != -1:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1  # reset and keep scanning

    raise ValueError(
        "Could not extract valid JSON from Claude response. "
        f"Response preview: {text[:300]!r}"
    )
