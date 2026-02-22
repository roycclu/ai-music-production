# Multi-Agent Music Video Production Pipeline

An AI-powered pipeline that orchestrates four specialized agents to produce a complete original music video — from a blank canvas to song, vocals, full production, and cinematic video scenes.

## Directory Structure

```
music_video_pipeline/
│
├── README.md
├── .env.example               # Copy to .env and fill in API keys
├── requirements.txt
│
├── main.py                    # Orchestrator — runs the full pipeline
│
├── agents/
│   ├── songwriter.py          # Agent 1: Original song, lyrics, chord chart
│   ├── singer.py              # Agent 2: Vocal direction + MiniMax music-01
│   ├── producer.py            # Agent 3: Production brief + MiniMax music-01
│   └── video_producer.py      # Agent 4: Visual treatment + MiniMax video-01
│
├── utils/
│   ├── claude_utils.py        # Shared Claude API helpers (streaming, JSON extraction)
│   ├── minimax_client.py      # Async MiniMax API client (music & video)
│   └── file_utils.py          # File I/O and directory constants
│
├── Melodies/                  # Agent 1 output: song structure, lyrics, chord chart
├── Vocals/                    # Agent 2 output: vocal direction, audio URL
├── Songs/                     # Agent 3 output: production brief, final track URL
├── Music Video/               # Agent 4 output: treatment, shot list, scene video URLs
│
└── final_deliverable.json     # Compiled production package (created on run)
```

## Pipeline Overview

```
Songwriter ──▶ Singer ──▶ Music Producer ──▶ Music Video Producer
     │              │              │                   │
  Melodies/      Vocals/        Songs/          Music Video/
```

| # | Agent | Claude role | MiniMax | Output folder |
|---|-------|------------|---------|---------------|
| 1 | Songwriter | Write song + chord chart | — | `Melodies/` |
| 2 | Singer | Vocal direction + MiniMax prompt | `music-01` (vocal track) | `Vocals/` |
| 3 | Music Producer | Production brief + MiniMax params | `music-01` (full track) | `Songs/` |
| 4 | Video Producer | Visual treatment + shot list | `video-01` (per scene) | `Music Video/` |

All Claude calls use `claude-opus-4-6` with **adaptive thinking** and streaming output.

## Setup

### 1. Install dependencies

```bash
cd music_video_pipeline
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
MINIMAX_API_KEY=...
MINIMAX_GROUP_ID=...     # required by some MiniMax subscription tiers
DEMO_MODE=false          # set true to test without MiniMax credits
```

**`DEMO_MODE=true`** runs the full Claude creative pipeline (Songwriter → Singer → Producer → Video Director) but replaces every MiniMax API call with a simulated response. Use this to test the pipeline structure and Claude outputs without spending MiniMax credits.

### 3. Run

```bash
python main.py
```

## Output Files

After a successful run you'll find:

| File | Contents |
|------|----------|
| `Melodies/song_structure.json` | Full song: title, lyrics (all sections), chord chart, BPM, key |
| `Melodies/lyrics.txt` | Formatted lyrics with section headers |
| `Melodies/chord_chart.txt` | Chord chart text |
| `Vocals/vocal_direction.md` | Performance direction document |
| `Vocals/minimax_params.json` | MiniMax music-01 parameters used |
| `Vocals/audio_url.txt` | Vocal track URL |
| `Songs/production_brief.md` | Full production brief |
| `Songs/minimax_params.json` | MiniMax music-01 parameters for full track |
| `Songs/audio_url.txt` | Final track URL |
| `Music Video/treatment.md` | Visual treatment and shot list |
| `Music Video/shot_list.json` | Scene-by-scene shot list |
| `Music Video/scene_NN_section.json` | Per-scene data including video URL |
| `Music Video/final_video_urls.json` | All video URLs in timeline order |
| `final_deliverable.json` | Complete production package |

## MiniMax API Notes

This pipeline targets the following MiniMax endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/music_generation` | Generate audio track (model: `music-01`) |
| `POST /v1/video_generation` | Initiate video clip generation (model: `video-01`) |
| `GET  /v1/query/video_generation?task_id=…` | Poll for video task completion |

Video generation is asynchronous. The client polls every 10 seconds up to 360 seconds per clip.

If your MiniMax subscription uses different endpoint paths or a different base URL, update `MINIMAX_BASE_URL` in your `.env` and the endpoint strings in `utils/minimax_client.py`.

## Customising the Song

To change the song's style, themes, or structure, edit the `USER_PROMPT` constant in `agents/songwriter.py`. All downstream agents automatically adapt to whatever the Songwriter produces.

## Architecture Notes

- **Async throughout**: All network I/O (Claude streaming + MiniMax polling) uses `asyncio` + `httpx`.
- **Retry logic**: Each MiniMax call retries once automatically on failure, with a refined prompt for video.
- **JSON extraction**: `utils/claude_utils.extract_json()` handles fenced code blocks and raw JSON objects from Claude's responses.
- **Demo mode**: No MiniMax key needed to test the creative pipeline. Set `DEMO_MODE=true`.
