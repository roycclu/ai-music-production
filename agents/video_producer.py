"""
Agent 4 — Music Video Producer
Creates a scene-by-scene visual treatment and generates each video clip
via MiniMax video-01.
Output folder: Music Video/
"""

import anthropic

from utils.claude_utils import call_claude, extract_json
from utils.file_utils import MUSIC_VIDEO_DIR, save_json, save_text
from utils.minimax_client import MinimaxClient

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an award-winning music video director known for intimate, \
cinematic, emotionally resonant work — films that feel like living inside a memory.

Your visual signature:
- Super 8 film grain, warm golden-hour light
- Summer afternoons, backyards, kids chasing each other
- Old family faces, weathered hands, open-mouthed laughter
- A single guitarist on a porch, dust motes in the air
- Nothing feels staged; everything feels true

When given a song, you produce a complete music video treatment: one visual scene \
per song section. Each scene prompt must be rich enough to guide an AI text-to-video \
model (MiniMax video-01) to generate the right images.

ALWAYS output a single JSON object in a ```json code block.

```json
{
  "treatment": "2–3 paragraph overarching visual treatment",
  "visual_style": "Specific cinematography and aesthetic notes",
  "color_palette": "Color palette and grade description",
  "scenes": [
    {
      "section":        "verse_1",
      "section_label":  "Verse 1",
      "prompt":         "Detailed MiniMax video-01 text prompt — cinematic, specific, evocative. At least 3 sentences.",
      "duration_seconds": 30,
      "camera_movement": "Slow push in from medium wide",
      "mood":           "Warm, nostalgic, quiet joy",
      "color_grade":    "Golden hour warmth, slight desaturation, 16mm grain"
    }
  ]
}
```

Create exactly one scene for each of these sections in order:
verse_1, pre_chorus, chorus, verse_2, bridge, outro

Make the prompts lush and specific — name the subjects, describe the light, \
describe the action, describe the feeling."""

# ── Agent entry point ──────────────────────────────────────────────────────────


async def run_video_producer(
    client: anthropic.AsyncAnthropic,
    song_data: dict,
    vocal_data: dict,
    production_data: dict,
) -> dict:
    """
    Agent 4: Write the video treatment and generate each scene via MiniMax video-01.

    Returns the video brief dict with scene_results including per-scene video URLs.
    """
    print("\n" + "═" * 60)
    print("  AGENT 4 — MUSIC VIDEO PRODUCER")
    print("  Creating visual treatment and generating video scenes…")
    print("═" * 60)

    lyrics = song_data.get("lyrics", {})
    brief = production_data.get("production_brief", {})

    # Build the lyrics summary for the prompt
    lyric_summary = _lyrics_summary(lyrics)

    user_content = f"""Using the complete song package below, create a music video \
treatment and a scene for each section.

SONG TITLE:   {song_data.get('title')}
KEY / BPM:    {song_data.get('key')} | {song_data.get('bpm')} BPM

LYRICS:
{lyric_summary}

PRODUCTION STYLE: {brief.get('style', '')}

DYNAMIC ARC:
{_arc_summary(brief.get('dynamic_arc', {}))}

CORE THEMES: Youth, joy, family, ache, the bittersweet beauty of time moving too fast.

VISUAL MANDATE: Super 8 film grain, golden-hour light, backyards, hands, laughter, \
a guitarist on a porch. Nothing staged. Everything true.

For each section (verse_1, pre_chorus, chorus, verse_2, bridge, outro) write a \
distinct visual scene. The chorus scene should feel the most expansive and joyful. \
The bridge should be the most raw and stripped. The outro should feel like letting go.

Make each "prompt" field detailed enough for MiniMax video-01 to generate a \
compelling 5–6 second clip that captures the scene's emotional truth.

Output as a single JSON object in a ```json code block."""

    response_text = await call_claude(
        client=client,
        system=SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=8192,
        label="Video Director → visual treatment + shot list",
    )

    video_data = extract_json(response_text)
    scenes = video_data.get("scenes", [])

    # ── Save treatment and shot list ───────────────────────────────────────────
    save_json(MUSIC_VIDEO_DIR / "video_brief.json", video_data)
    save_text(MUSIC_VIDEO_DIR / "treatment.md", _format_treatment_md(video_data, song_data))
    save_json(MUSIC_VIDEO_DIR / "shot_list.json", scenes)

    # ── Generate each scene via MiniMax video-01 ───────────────────────────────
    print(f"\n[Agent 4] Generating {len(scenes)} video scene(s) via MiniMax video-01…\n")
    minimax = MinimaxClient()
    scene_results = []

    for idx, scene in enumerate(scenes, start=1):
        section = scene.get("section", f"scene_{idx}")
        label = scene.get("section_label", section)
        prompt = scene.get("prompt", "")
        desired_duration = scene.get("duration_seconds", 5)

        print(f"  Scene {idx}/{len(scenes)}: {label}")
        print(f"  Prompt excerpt: {prompt[:80]}…")

        # MiniMax video-01 typically caps individual clip duration at ~5–6 s.
        # We request the minimum of desired_duration and 6 s per clip.
        clip_duration = min(desired_duration, 6)

        video_result = await minimax.generate_video(
            prompt=prompt,
            duration=clip_duration,
        )

        video_url = (
            video_result.get("video_url")
            or video_result.get("download_url")
            or video_result.get("file_url")
            or "N/A"
        )

        scene_result = {
            "scene_number":   idx,
            "section":        section,
            "section_label":  label,
            "prompt":         prompt,
            "duration_seconds": desired_duration,
            "clip_duration":  clip_duration,
            "camera_movement": scene.get("camera_movement", ""),
            "mood":           scene.get("mood", ""),
            "color_grade":    scene.get("color_grade", ""),
            "video_url":      video_url,
        }
        scene_results.append(scene_result)

        safe_section = section.replace(" ", "_")
        save_json(MUSIC_VIDEO_DIR / f"scene_{idx:02d}_{safe_section}.json", scene_result)
        save_text(MUSIC_VIDEO_DIR / f"scene_{idx:02d}_{safe_section}_url.txt", video_url)
        print(f"  → {video_url}\n")

    # ── Save consolidated video URL manifest ───────────────────────────────────
    url_manifest = {
        "song_title":   song_data.get("title"),
        "total_scenes": len(scene_results),
        "timeline": [
            {
                "scene":    s["scene_number"],
                "section":  s["section_label"],
                "duration": s["duration_seconds"],
                "url":      s["video_url"],
            }
            for s in scene_results
        ],
    }
    save_json(MUSIC_VIDEO_DIR / "final_video_urls.json", url_manifest)

    print(f"[Agent 4 ✓] {len(scene_results)} scenes generated.\n")
    return {
        **video_data,
        "scenes": scene_results,
        "video_urls": [s["video_url"] for s in scene_results],
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _lyrics_summary(lyrics: dict) -> str:
    section_order = [
        ("verse_1",    "Verse 1"),
        ("pre_chorus", "Pre-Chorus"),
        ("chorus",     "Chorus"),
        ("verse_2",    "Verse 2"),
        ("bridge",     "Bridge"),
        ("outro",      "Outro"),
    ]
    parts = []
    for key, label in section_order:
        text = lyrics.get(key, "").strip()
        if text:
            # Just the first two lines per section to keep prompt concise
            snippet = "\n".join(text.splitlines()[:2])
            parts.append(f"[{label}]\n{snippet}")
    return "\n\n".join(parts)


def _arc_summary(arc: dict) -> str:
    return "\n".join(f"  {k.title()}: {v}" for k, v in arc.items())


def _format_treatment_md(video_data: dict, song_data: dict) -> str:
    title = song_data.get("title", "Untitled")
    scenes = video_data.get("scenes", [])

    lines = [
        f"# Music Video Treatment: {title}",
        "",
        "## Overview",
        video_data.get("treatment", ""),
        "",
        "## Visual Style",
        video_data.get("visual_style", ""),
        "",
        "## Color Palette",
        video_data.get("color_palette", ""),
        "",
        "## Shot List",
        "",
    ]
    for i, scene in enumerate(scenes, start=1):
        label = scene.get("section_label", scene.get("section", f"Scene {i}"))
        lines += [
            f"### Scene {i}: {label}",
            f"**Duration:** {scene.get('duration_seconds', 5)} seconds  ",
            f"**Camera:** {scene.get('camera_movement', '')}  ",
            f"**Mood:** {scene.get('mood', '')}  ",
            f"**Color Grade:** {scene.get('color_grade', '')}",
            "",
            f"**Prompt:**  ",
            scene.get("prompt", ""),
            "",
        ]
    return "\n".join(lines)
