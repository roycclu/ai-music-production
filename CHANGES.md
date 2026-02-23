# Changes

## 2026-02-23
Added Arize Phoenix tracing (utils/observability.py). All Anthropic API calls are now traced to the 'music-video-pipeline' project at http://localhost:6006. setup_observability() is called once at pipeline startup in main.py before the Anthropic client is created.

## 2026-02-22
Init: Added git version control, CHANGES.md session log, and smart cache layer. Pipeline now skips re-generating Melodies/Vocals/Songs/Video when output files already exist. Pass `--force` to regenerate everything from scratch.

Added prompt caching: each agent saves system_prompt.txt + user_prompt.txt in its output folder. On re-run, prompts are loaded from disk instead of reconstructed from prior agents' data.

Added sub-caching within agents 2, 3, 4: if the Claude-generated output (vocal_direction.json / production_brief.json / video_brief.json) already exists, the Claude call is skipped and only the media generation step runs. Agent 4 also caches per-scene video results — already-generated scenes are skipped on re-run. Delete specific files to selectively regenerate.

Added ElevenLabs as music generation fallback (utils/elevenlabs_client.py + utils/music_generator.py). Agents 2 and 3 now try MiniMax first; if MiniMax raises an error, ElevenLabs TTS is used automatically. Pipeline only halts if both providers fail. Set ELEVENLABS_API_KEY in .env to enable. Banner now shows active provider config at startup.
