# Development guide

## v1.0 beta

Local read-aloud for long text (URL fetch + paste) with MLX Qwen3 CustomVoice on Apple Silicon.

## Layout

| File | Role |
|------|------|
| `read_aloud_server.py` | Flask app, sessions, synthesis queue |
| `tts_engine.py` | MLX catalog, download, synthesis |
| `read_aloud.py` | CLI + text chunking |
| `url_fetcher.py` | Page text extraction |
| `static/app.js` | Playback, highlight sync, chunk status |
| `templates/index.html` | Main + sidebar layout |

## APIs

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/status` | Model state |
| GET | `/api/models` | Model catalog |
| POST | `/api/models/select` | Switch model |
| POST | `/api/models/download` | Start download |
| GET | `/api/voices` | Speakers |
| POST | `/api/fetch` | URL → text |
| POST | `/api/prepare` | Create session |
| GET | `/api/chunk/<session>/<index>` | Chunk WAV |
| GET | `/api/session/<session>/status` | Cache / synth state |
| POST | `/api/session/<session>/stop` | Cancel session |

## Grok resume

Grok stores sessions under `~/.grok/sessions/<encoded-project-path>/`. Use `/resume` in Grok to continue a prior session for this folder.

Portable context in git: `AGENTS.md`, this file, `.grok/skills/read-aloud-dev/`.

## Publish to GitHub

```bash
git init
git add -A
git commit -m "v1.0 beta: MLX Read Aloud"
git tag v1.0.0-beta.1
git remote add origin git@github.com:YOU/read-aloud.git
git push -u origin main
git push origin v1.0.0-beta.1
```

## Follow-ups

- Pre-synthesize first chunk on prepare
- Cancel in-flight MLX generation mid-chunk
- Support VoiceDesign / Base model types in UI