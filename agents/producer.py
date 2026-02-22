"""
Agent 3 — Music Producer
Creates a detailed production brief and generates the fully produced
track via MiniMax music-01.
Output folder: Songs/
"""

import anthropic

from utils.claude_utils import call_claude, extract_json
from utils.file_utils import SONGS_DIR, format_lyrics_with_markers, load_prompt, save_json, save_prompt, save_text
from utils.minimax_client import MinimaxClient

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a world-class music producer who fuses acoustic folk \
authenticity with progressive-electronic architecture — Avicii's emotional chord \
progressions and anthemic builds married to Sufjan Stevens' intimacy and warmth.

Your production philosophy:
- Simple, repeatable beat loops that anchor without overpowering
- Arrangements that crescendo from intimate acoustic verse to full anthemic chorus
- Emotional resolution: joy, ache, and finally peace
- Instrumentation palette: acoustic guitar, banjo, kick/snare, synth pads, bass, strings

When given a song's chord chart, BPM, key, and vocal direction, you produce a \
complete production brief and MiniMax music-01 API parameters.

ALWAYS output a single JSON object in a ```json code block.

```json
{
  "production_brief": {
    "style": "One-paragraph production style description",
    "instrumentation": ["acoustic guitar", "banjo", "kick drum", "..."],
    "dynamic_arc": {
      "verse":      "Intimate — guitar and subtle pad only",
      "pre_chorus": "Building — hi-hat enters, bass creeps in",
      "chorus":     "Full — all instruments, kick drives the drop",
      "bridge":     "Stripped — voice and guitar alone",
      "outro":      "Gentle fade — instruments fall away one by one"
    },
    "mixing_notes": "Reverb, compression, EQ, and stereo field notes",
    "tempo_feel":   "Groove and feel description (e.g. 'laid-back quarter-note pulse')"
  },
  "minimax_api_params": {
    "model":            "music-01",
    "lyrics":           "Full lyrics with [Section] markers",
    "vocal_style":      "Vocal style carried forward from singer agent",
    "genre":            "folk-electronic",
    "mood":             "bittersweet-anthemic",
    "bpm":              92,
    "key":              "G major",
    "instruments":      ["acoustic guitar", "banjo", "synthesizer pads", "kick drum", "strings"],
    "production_style": "Avicii-folk fusion — intimate verses, euphoric chorus drops"
  }
}
```"""

# ── Agent entry point ──────────────────────────────────────────────────────────


async def run_producer(
    client: anthropic.AsyncAnthropic,
    song_data: dict,
    vocal_data: dict,
) -> dict:
    """
    Agent 3: Write a production brief and generate the full track via MiniMax.

    Returns production_brief dict plus audio_url from MiniMax.
    """
    print("\n" + "═" * 60)
    print("  AGENT 3 — MUSIC PRODUCER")
    print("  Writing production brief and generating final track…")
    print("═" * 60)

    vocal_direction = vocal_data.get("vocal_direction", {})
    lyrics_block = format_lyrics_with_markers(song_data)

    # ── Prompt cache ───────────────────────────────────────────────────────────
    cached_user = load_prompt(SONGS_DIR / "user_prompt.txt")
    if cached_user is not None:
        user_content = cached_user
        print("  [Prompt] Loaded from Songs/user_prompt.txt")
    else:
        user_content = f"""Using the complete song package below, write a production brief \
and build the MiniMax music-01 API parameters for the fully produced track.

SONG TITLE:  {song_data.get('title')}
KEY: {song_data.get('key')}  |  BPM: {song_data.get('bpm')}  |  TIME: {song_data.get('time_signature')}

CHORD CHART:
{song_data.get('chord_chart', '')}

VOCAL STYLE:   {vocal_direction.get('style', '')}
LEAD VOICE:    {vocal_direction.get('lead_voice', '')}
HARMONIES:     {vocal_direction.get('harmonies', '')}
DYNAMICS:      {vocal_direction.get('dynamics', '')}

SONGWRITER NOTES:
{song_data.get('notes', '')}

LYRICS:
{lyrics_block}

Specify the full production arc section-by-section, all instrumentation, and \
mixing priorities. The chorus should feel like an Avicii-style euphoric build \
releasing into something tender — bluegrass authenticity meets progressive house emotion.

Output as a single JSON object in a ```json code block."""
        save_prompt(SONGS_DIR / "system_prompt.txt", SYSTEM_PROMPT)
        save_prompt(SONGS_DIR / "user_prompt.txt", user_content)

    response_text = await call_claude(
        client=client,
        system=SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=8192,
        label="Producer → production brief + MiniMax API params",
    )

    production_data = extract_json(response_text)

    # ── Persist outputs ────────────────────────────────────────────────────────
    save_json(SONGS_DIR / "production_brief.json", production_data)
    save_text(SONGS_DIR / "production_brief.md", _format_brief_md(production_data, song_data))
    save_json(SONGS_DIR / "minimax_params.json", production_data.get("minimax_api_params", {}))

    # ── MiniMax music-01 call ──────────────────────────────────────────────────
    print("\n[Agent 3] Calling MiniMax music-01 for full track generation…")
    minimax = MinimaxClient()
    api_params = production_data.get("minimax_api_params", {})

    music_result = await minimax.generate_music(
        lyrics=api_params.get("lyrics", lyrics_block),
        vocal_style=api_params.get(
            "vocal_style",
            vocal_direction.get("style", "soulful folk"),
        ),
        genre=api_params.get("genre", "folk-electronic"),
        bpm=song_data.get("bpm"),
        key=song_data.get("key"),
    )

    audio_url = (
        music_result.get("audio_url")
        or music_result.get("download_url")
        or "N/A"
    )

    save_json(SONGS_DIR / "audio_result.json", music_result)
    save_text(SONGS_DIR / "audio_url.txt", audio_url)

    print(f"\n[Agent 3 ✓] Final track URL: {audio_url}\n")
    return {**production_data, "audio_url": audio_url, "audio_result": music_result}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_brief_md(production_data: dict, song_data: dict) -> str:
    brief = production_data.get("production_brief", {})
    title = song_data.get("title", "Untitled")

    lines = [
        f"# Production Brief: {title}",
        f"*{song_data.get('key')} | {song_data.get('bpm')} BPM | {song_data.get('time_signature')}*",
        "",
        "## Production Style",
        brief.get("style", ""),
        "",
        "## Instrumentation",
        "",
    ]
    for instrument in brief.get("instrumentation", []):
        lines.append(f"- {instrument}")
    lines += ["", "## Dynamic Arc", ""]
    for section, description in brief.get("dynamic_arc", {}).items():
        lines.append(f"**{section.title()}:** {description}")
    lines += [
        "",
        "## Mixing Notes",
        brief.get("mixing_notes", ""),
        "",
        "## Tempo & Feel",
        brief.get("tempo_feel", ""),
    ]
    return "\n".join(lines)
