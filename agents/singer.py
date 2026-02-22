"""
Agent 2 — Singer / Vocal Director
Creates a detailed vocal performance direction document and generates
the vocal track via MiniMax music-01.
Output folder: Vocals/
"""

import anthropic

from utils.claude_utils import call_claude, extract_json
from utils.file_utils import VOCALS_DIR, format_lyrics_with_markers, load_prompt, save_json, save_prompt, save_text
from utils.minimax_client import MinimaxClient

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert vocal director and recording producer with deep \
knowledge of vocal performance, studio technique, and AI music generation APIs.

Your job is twofold:
1. Write a granular vocal performance direction document describing HOW the song \
   should be sung — style, phrasing, emotional delivery, dynamics, and harmonies.
2. Construct the exact MiniMax music-01 API parameters that capture this vocal vision.

VOCAL CHARACTER
- Lead voice: raw, aching bluegrass tenor (Dan Tyminski) fused with soulful, \
  warm delivery (Aloe Blacc). Deeply human, earnest, emotionally present.
- Chorus harmonies: intimate family choir feeling — 2–3 part stack, not polished, \
  slightly imperfect, full of warmth.

ALWAYS output a single JSON object in a ```json code block.

```json
{
  "vocal_direction": {
    "style": "One-paragraph overall style description",
    "lead_voice": "Specific lead voice characteristics",
    "emotional_delivery": "How to deliver each emotional beat",
    "phrasing": "Phrasing, breath placement, articulation",
    "dynamics": "Dynamic range and variation across the song",
    "harmonies": "Harmony arrangement for choruses and outros",
    "section_notes": {
      "verse_1":    "Specific direction for verse 1",
      "pre_chorus": "Specific direction for pre-chorus",
      "chorus":     "Specific direction for chorus",
      "verse_2":    "Specific direction for verse 2",
      "bridge":     "Specific direction for bridge",
      "outro":      "Specific direction for outro"
    }
  },
  "minimax_prompt": {
    "lyrics": "Full lyrics with [Verse 1] [Pre-Chorus] [Chorus] etc. markers",
    "vocal_style": "Concise vocal style string for MiniMax API",
    "genre": "folk-soul",
    "mood": "bittersweet-hopeful",
    "additional_params": {
      "voice_character": "Brief voice character note",
      "harmony_style":   "Harmony note"
    }
  }
}
```"""

# ── Agent entry point ──────────────────────────────────────────────────────────


async def run_singer(client: anthropic.AsyncAnthropic, song_data: dict) -> dict:
    """
    Agent 2: Generate vocal direction and create the vocal track via MiniMax.

    Returns the vocal_direction dict plus audio_url from MiniMax.
    """
    print("\n" + "═" * 60)
    print("  AGENT 2 — SINGER / VOCAL DIRECTOR")
    print("  Creating vocal performance direction and generating vocals…")
    print("═" * 60)

    lyrics_block = format_lyrics_with_markers(song_data)

    # ── Prompt cache ───────────────────────────────────────────────────────────
    cached_user = load_prompt(VOCALS_DIR / "user_prompt.txt")
    if cached_user is not None:
        user_content = cached_user
        print("  [Prompt] Loaded from Vocals/user_prompt.txt")
    else:
        user_content = f"""Using the song below, write a detailed vocal performance direction \
document and construct the MiniMax music-01 API parameters.

SONG TITLE: {song_data.get('title')}
KEY: {song_data.get('key')}  |  BPM: {song_data.get('bpm')}  |  TIME: {song_data.get('time_signature')}

LYRICS:
{lyrics_block}

CHORD CHART:
{song_data.get('chord_chart', '')}

SONGWRITER NOTES:
{song_data.get('notes', '')}

Provide section-by-section vocal direction. In the minimax_prompt.lyrics field, \
include the complete lyrics with proper [Section] markers so MiniMax knows \
where each section begins.

Output as a single JSON object in a ```json code block."""
        save_prompt(VOCALS_DIR / "system_prompt.txt", SYSTEM_PROMPT)
        save_prompt(VOCALS_DIR / "user_prompt.txt", user_content)

    response_text = await call_claude(
        client=client,
        system=SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=8192,
        label="Singer → vocal direction + MiniMax prompt",
    )

    vocal_data = extract_json(response_text)

    # ── Persist outputs ────────────────────────────────────────────────────────
    save_json(VOCALS_DIR / "vocal_direction.json", vocal_data)
    save_text(VOCALS_DIR / "vocal_direction.md", _format_direction_md(vocal_data, song_data))
    save_json(VOCALS_DIR / "minimax_params.json", vocal_data.get("minimax_prompt", {}))

    # ── MiniMax music-01 call ──────────────────────────────────────────────────
    print("\n[Agent 2] Calling MiniMax music-01 for vocal track generation…")
    minimax = MinimaxClient()
    mp = vocal_data.get("minimax_prompt", {})

    music_result = await minimax.generate_music(
        lyrics=mp.get("lyrics", lyrics_block),
        vocal_style=mp.get("vocal_style", "soulful folk tenor with aching bluegrass warmth"),
        genre=mp.get("genre", "folk-soul"),
        bpm=song_data.get("bpm"),
        key=song_data.get("key"),
    )

    audio_url = (
        music_result.get("audio_url")
        or music_result.get("download_url")
        or "N/A"
    )

    save_json(VOCALS_DIR / "audio_result.json", music_result)
    save_text(VOCALS_DIR / "audio_url.txt", audio_url)

    print(f"\n[Agent 2 ✓] Vocal track URL: {audio_url}\n")
    return {**vocal_data, "audio_url": audio_url, "audio_result": music_result}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_direction_md(vocal_data: dict, song_data: dict) -> str:
    direction = vocal_data.get("vocal_direction", {})
    title = song_data.get("title", "Untitled")

    lines = [
        f"# Vocal Direction: {title}",
        "",
        "## Overall Style",
        direction.get("style", ""),
        "",
        "## Lead Voice",
        direction.get("lead_voice", ""),
        "",
        "## Emotional Delivery",
        direction.get("emotional_delivery", ""),
        "",
        "## Phrasing & Articulation",
        direction.get("phrasing", ""),
        "",
        "## Dynamics",
        direction.get("dynamics", ""),
        "",
        "## Harmonies",
        direction.get("harmonies", ""),
        "",
        "## Section-by-Section Notes",
        "",
    ]
    for section, notes in direction.get("section_notes", {}).items():
        label = section.replace("_", " ").title()
        lines += [f"### {label}", notes, ""]

    return "\n".join(lines)
