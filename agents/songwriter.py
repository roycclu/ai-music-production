"""
Agent 1 — Songwriter
Writes an original song: title, full lyrics (6 sections), chord chart,
BPM, key, and time signature.
Output folder: Melodies/
"""

import anthropic

from utils.claude_utils import call_claude, extract_json
from utils.file_utils import MELODIES_DIR, load_prompt, save_json, save_prompt, save_text

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional songwriter with deep expertise in folk, \
Americana, and soulful acoustic music. You write songs that feel cinematic and \
emotionally resonant — songs that capture memories and make them feel alive.

When you write a song you deliver:
1. A compelling, evocative song title
2. Full lyrics with clearly labelled sections
3. A complete chord chart with performance notes
4. Musical metadata: BPM, key, time signature

ALWAYS output your response as a single JSON object inside a ```json code block. \
Do not include explanatory prose outside the code block.

Use this exact JSON structure:

```json
{
  "title": "Song Title Here",
  "lyrics": {
    "verse_1": "First verse lyrics — each line on its own line",
    "pre_chorus": "Pre-chorus lyrics",
    "chorus": "Chorus lyrics",
    "verse_2": "Second verse lyrics",
    "bridge": "Bridge lyrics",
    "outro": "Outro lyrics"
  },
  "chord_chart": "Full chart — list each section with chord names and timing",
  "bpm": 92,
  "key": "G major",
  "time_signature": "4/4",
  "notes": "Performance feel, guitarist instructions, song arc notes"
}
```"""

# ── User prompt ────────────────────────────────────────────────────────────────

USER_PROMPT = """Write a complete original song with these qualities:

FEEL & MOOD
- Hopeful, soulful, guitar-based
- Half melancholy, half bursting with youth and living in the moment
- The bittersweet ache of not wanting beautiful moments to pass

THEMES
- Youth, joy, and family
- Time moving too fast
- A memory being lived in real time

SONG STRUCTURE (in order)
Verse 1 → Pre-Chorus → Chorus → Verse 2 → Pre-Chorus → Chorus → Bridge → Outro

TARGET DURATION: 3.5 – 4 minutes at the chosen BPM

VOCAL STYLE HINT (for chord/key choices)
The lead will blend Dan Tyminski's raw bluegrass tenor with Aloe Blacc's soulful warmth.
Write in a key that suits a warm, aching mid-tenor voice.

Output the complete song as a JSON object in a ```json code block."""

# ── Agent entry point ──────────────────────────────────────────────────────────


async def run_songwriter(client: anthropic.AsyncAnthropic) -> dict:
    """
    Agent 1: Generate the original song.

    Returns a dict with keys:
      title, lyrics (dict of sections), chord_chart, bpm, key, time_signature, notes
    """
    print("\n" + "═" * 60)
    print("  AGENT 1 — SONGWRITER")
    print("  Writing original song with lyrics and chord structure…")
    print("═" * 60)

    # ── Prompt cache ───────────────────────────────────────────────────────────
    # Songwriter prompts are static constants, but we persist them to disk so
    # they can be inspected and (if desired) hand-edited before a --force rerun.
    cached_user = load_prompt(MELODIES_DIR / "user_prompt.txt")
    user_to_send = cached_user if cached_user is not None else USER_PROMPT
    if cached_user is None:
        save_prompt(MELODIES_DIR / "system_prompt.txt", SYSTEM_PROMPT)
        save_prompt(MELODIES_DIR / "user_prompt.txt", USER_PROMPT)
    else:
        print("  [Prompt] Loaded from Melodies/user_prompt.txt")

    response_text = await call_claude(
        client=client,
        system=SYSTEM_PROMPT,
        user_content=user_to_send,
        max_tokens=8192,
        label="Songwriter → generating song",
    )

    song_data = extract_json(response_text)

    # ── Persist outputs ────────────────────────────────────────────────────────
    save_json(MELODIES_DIR / "song_structure.json", song_data)
    save_text(MELODIES_DIR / "lyrics.txt", _formatted_lyrics(song_data))
    save_text(MELODIES_DIR / "chord_chart.txt", song_data.get("chord_chart", ""))

    title = song_data.get("title", "Untitled")
    key = song_data.get("key", "?")
    bpm = song_data.get("bpm", "?")
    print(f"\n[Agent 1 ✓] '{title}' — {key}, {bpm} BPM\n")

    return song_data


# ── Helpers ────────────────────────────────────────────────────────────────────

def _formatted_lyrics(song_data: dict) -> str:
    title = song_data.get("title", "Untitled")
    key = song_data.get("key", "?")
    bpm = song_data.get("bpm", "?")
    time_sig = song_data.get("time_signature", "4/4")
    lyrics = song_data.get("lyrics", {})

    section_order = [
        ("verse_1",    "VERSE 1"),
        ("pre_chorus", "PRE-CHORUS"),
        ("chorus",     "CHORUS"),
        ("verse_2",    "VERSE 2"),
        ("pre_chorus", "PRE-CHORUS"),
        ("chorus",     "CHORUS"),
        ("bridge",     "BRIDGE"),
        ("outro",      "OUTRO"),
    ]

    lines = [
        f"TITLE: {title}",
        f"KEY: {key}  |  BPM: {bpm}  |  TIME: {time_sig}",
        "=" * 60,
        "",
    ]
    for key_name, label in section_order:
        text = lyrics.get(key_name, "").strip()
        if text:
            lines += [f"[{label}]", text, ""]

    return "\n".join(lines)
