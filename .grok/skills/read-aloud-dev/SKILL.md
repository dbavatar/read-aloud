---
name: read-aloud-dev
description: >
  Develop and debug the Read Aloud MLX TTS project (read_aloud_server.py,
  tts_engine.py, static/app.js). Use for playback, synthesis, UI, or new features.
---

# Read Aloud development

1. Read `AGENTS.md`, `DESIGN.md`, and `DEVELOPMENT.md`.
2. Run: `source venv/bin/activate && python read_aloud_server.py --no-browser`
3. MLX synthesis only on `synthesis_worker` — queue via `SYNTH_QUEUE`.
4. Do not commit `venv/` or model weights (HF cache is under `~/.cache/huggingface`).

Smoke test:

```python
from read_aloud_server import app
with app.test_client() as c:
    r = c.post("/api/prepare", json={"text": "Hello.", "speaker": "ryan"})
    print(r.status_code, r.json)
```