# Read Aloud

Local text-to-speech reader for **Apple Silicon**, built on [mlx-audio](https://github.com/Blaizzy/mlx-audio) and Qwen3-TTS. Fetch webpages or paste text, then listen with speed controls and synced highlighting.

**v1.0 beta**

## Requirements

- macOS with Apple Silicon (M-series)
- Python 3.12

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On first use, the default MLX model (~2.4 GB) downloads from Hugging Face automatically.

## Run

```bash
source venv/bin/activate
python read_aloud_server.py
```

Open the URL printed in the terminal (default: `http://127.0.0.1:8765`).

Optional CLI:

```bash
python read_aloud.py --text "Hello world" --speaker ryan
```

## Features

- Webpage URL fetch or paste text
- MLX Qwen3 CustomVoice voices
- Play / pause / stop with synthesis cancel
- 2× default playback speed
- Steady audiobook tone across chunks
- Chunk highlighting synced to audio
- Model download and switch in the UI

## Development

- `AGENTS.md` — agent/Grok project rules (auto-loaded)
- `DESIGN.md` — architecture decisions and compacted beta history
- `DEVELOPMENT.md` — file map and APIs
- `.grok/skills/read-aloud-dev/` — Grok skill for this repo

## License

Apache 2.0 — see [LICENSE](LICENSE).