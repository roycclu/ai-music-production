# Changes

## 2026-02-22
Init: Added git version control, CHANGES.md session log, and smart cache layer. Pipeline now skips re-generating Melodies/Vocals/Songs/Video when output files already exist. Pass `--force` to regenerate everything from scratch.

Added prompt caching: each agent saves system_prompt.txt + user_prompt.txt in its output folder. On re-run, prompts are loaded from disk instead of reconstructed from prior agents' data.
