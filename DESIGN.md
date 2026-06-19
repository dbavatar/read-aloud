# Design notes (v1.0 beta)

Compacted context for resuming development. No dependency on external session logs.

## Goal

Local TTS for **long text** (fetched pages or pasted content) on Apple Silicon: web GUI, chunked playback, speed control, synced highlighting, model management.

## Backend choice

- **Shipped:** MLX via `mlx-audio` + Qwen3-TTS CustomVoice models.
- **Not used:** LM Studio for TTS. Qwen3 TTS entries in LM Studio fail to load; there is no audio speech API. Orpheus GGUF + LM Studio is a separate legacy approach (out of scope for this repo).

Default model: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` (~2.4 GB, Hugging Face cache).

## Threading model (critical)

MLX GPU context is **thread-bound**. All synthesis must run on one worker:

```
Flask threads → SYNTH_QUEUE → synthesis_worker() → synthesize_chunk()
```

Never call `load_model` / `generate_custom_voice` from Flask request handlers. Model reload jobs use the same queue (`type: "reload"`).

## Session lifecycle

1. `POST /api/prepare` — chunk text (~600 chars), return `session_id` + chunk list.
2. `GET /api/chunk/<session>/<index>` — return cached WAV or synthesize.
3. `POST /api/session/<id>/stop` — set `cancelled`; worker skips queued jobs for that session.

Server tracks `synthesizing_index` and `audio_cache` per session.

## Tone stability across chunks

Each chunk is synthesized independently; defaults caused audible tone shifts at boundaries.

Mitigations in `tts_engine.py`:

- `STEADY_READING_INSTRUCT` — calm audiobook style on every chunk (UI toggle, default on).
- `temperature=0.65`, `top_p=0.9` (down from 0.9 / 1.0).
- Larger chunks (600 chars) → fewer boundaries.

## Client playback

Single active loop guarded by `playAllLock` + `playAllPromise`.

**Stop / cancel:**

- Increment `playbackGeneration` to invalidate in-flight work.
- `AbortController` on chunk HTTP fetches.
- Server session cancel stops queued synthesis.

**Highlight sync at 2× speed:**

- Browser `audio.duration` is unreliable for blob WAV → decode duration via Web Audio API.
- Progress uses `max(currentTime/duration, wallClockElapsed)` at current `playbackRate`.
- `requestAnimationFrame` for cursor; force ratio `1.0` on chunk end.
- Default playback speed: **2×** (browser `playbackRate`, not server resample).

**Chunk UI states:** pending → generating → ready (cached) → playing → done. Sidebar pipeline text + colored borders in reading view.

## URL fetch

`url_fetcher.py` uses trafilatura only. Sites that block bots or need login: user pastes text manually. No platform-specific fetchers.

## UI layout (beta)

- Main column: hero, URL input, text panel, model panel (bottom).
- Sticky sidebar: voice, speed, steady tone, transport, progress, chunk legend.

## Bugs fixed during beta (do not regress)

| Issue | Cause | Fix |
|-------|-------|-----|
| "Preparing…" forever | `SESSION_LOCK` re-entry in TTL cleanup | `session_ttl_cleanup_locked()` |
| MLX segfault | Model/synth on different threads | Dedicated worker + queue |
| Stop ignored | Client loop + server queue continued | `playbackGeneration`, abort fetches, session cancel |
| Highlight lag / stops early | Wrong duration, throttled updates | Web Audio decode + wall clock + rAF |
| Double Play races | Concurrent `playAll()` | `playAllLock` |

## Known limits

- One MLX job at a time; chunk in progress may finish after Stop (no mid-generation interrupt).
- Highlight uses linear char mapping vs. speech timing (good enough; snap to 100% on end).
- VoiceDesign / Base / clone model types listed but server rejects non-CustomVoice for read-aloud.
- LM Studio MLX copy may lack `speech_tokenizer`; engine prefers complete HF cache snapshot.

## Grok / agent resume

| Source | In git? | Role |
|--------|---------|------|
| `AGENTS.md` | Yes | Auto-loaded rules, stack, conventions |
| `DEVELOPMENT.md` | Yes | File map, APIs, publish steps |
| `DESIGN.md` | Yes | Decisions, constraints, history (this file) |
| `.grok/skills/read-aloud-dev/` | Yes | Task skill for edits |
| `~/.grok/sessions/...` | No | Full chat log; use `/resume` locally if needed |

Clone + read `AGENTS.md` and `DESIGN.md` is sufficient to continue building without the original session.

## Post-beta ideas

- Pre-synthesize chunk 0 on prepare
- Cooperative cancel inside MLX generation
- Server-side speed (resample) as alternative to browser rate
- VoiceDesign / voice clone flows