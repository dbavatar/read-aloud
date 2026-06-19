# Read Aloud — agent context

Local TTS reader for Apple Silicon using **MLX + Qwen3-TTS**.

## Stack

- **Backend:** `mlx-audio` via `tts_engine.py` (dedicated synthesis worker thread)
- **Server:** `read_aloud_server.py` — Flask web GUI
- **UI:** `templates/index.html`, `static/app.js`, `static/style.css`
- **CLI:** `read_aloud.py`
- **Fetch:** `url_fetcher.py` (trafilatura; paste fallback when fetch fails)

## Run

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python read_aloud_server.py
```

Default model: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`.

## Architecture

- MLX runs on a **single worker thread** (`synthesis_worker` + `SYNTH_QUEUE`).
- Sessions: `POST /api/prepare` → chunks; `GET /api/chunk/<session>/<index>` synthesizes on demand.
- `POST /api/session/<id>/stop` cancels queued synthesis.
- Client uses `playbackGeneration` + `AbortController` for reliable stop.
- Highlight sync decodes WAV duration via Web Audio API.

## Conventions

- Python 3.12 venv at `venv/`
- Do not commit venvs or generated audio
- See `DEVELOPMENT.md` for APIs and follow-ups