#!/usr/bin/env python3
"""Local web GUI for reading webpages and pasted text aloud."""

from __future__ import annotations

import argparse
import io
import queue
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_file

from read_aloud import chunk_text, clean_text, playback_position_from_offset
from tts_engine import (
    DEFAULT_MODEL,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    READ_ALOUD_VOICE_TYPES,
    STEADY_READING_INSTRUCT,
    STEADY_READING_TEMPERATURE,
    STEADY_READING_TOP_P,
    catalog_entry,
    chunk_offsets,
    download_model,
    get_backend_info,
    list_models,
    list_voices,
    load_model,
    model_download_status,
    configure_tts_runtime,
    resolve_speaker,
    set_current_model,
    steady_chunk_instruct,
    synthesize_chunk,
)
from build_info import get_build_info
from url_fetcher import fetch_url

app = Flask(__name__)

MODEL = None
MODEL_READY = threading.Event()
MODEL_LOADING = threading.Event()
ACTIVE_MODEL_ID = DEFAULT_MODEL
SYNTH_QUEUE: queue.Queue[dict[str, Any] | None] = queue.Queue()
SESSIONS: dict[str, "ReadSession"] = {}
SESSION_LOCK = threading.Lock()


@dataclass
class ReadSession:
    chunks: list[str]
    chunk_starts: list[int]
    full_text: str
    speaker: str
    language: str = "English"
    instruct: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    cancelled: bool = False
    synthesizing_index: int | None = None
    audio_cache: dict[int, bytes] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


def session_is_cancelled(session_id: str | None) -> bool:
    if not session_id:
        return False
    with SESSION_LOCK:
        session = SESSIONS.get(session_id)
        return session is not None and session.cancelled


def cancel_session(session_id: str) -> bool:
    with SESSION_LOCK:
        session = SESSIONS.get(session_id)
        if session is None:
            return False
        session.cancelled = True
        return True


def synthesis_worker() -> None:
    global MODEL, ACTIVE_MODEL_ID

    while True:
        job = SYNTH_QUEUE.get()
        if job is None:
            break

        if job.get("type") == "reload":
            previous_model = MODEL
            previous_id = ACTIVE_MODEL_ID
            MODEL_READY.clear()
            MODEL_LOADING.set()
            try:
                ACTIVE_MODEL_ID = job["model_id"]
                set_current_model(ACTIVE_MODEL_ID)
                MODEL = load_model(ACTIVE_MODEL_ID)
                MODEL_READY.set()
                job["error"] = None
                print(f"MLX TTS model ready: {ACTIVE_MODEL_ID}")
            except Exception as exc:
                # Keep the previous model live so a failed switch does not
                # leave the daemon permanently unready for playback.
                if previous_model is not None:
                    MODEL = previous_model
                    ACTIVE_MODEL_ID = previous_id
                    set_current_model(previous_id)
                    MODEL_READY.set()
                    print(f"Model load failed; restored {previous_id}: {exc}")
                else:
                    MODEL = None
                    print(f"Model load failed: {exc}")
                job["error"] = exc
            finally:
                MODEL_LOADING.clear()
                job["event"].set()
            continue

        if not MODEL_READY.is_set() or MODEL is None:
            job["error"] = RuntimeError("TTS model is not ready")
            job["event"].set()
            continue

        if session_is_cancelled(job.get("session_id")):
            job["error"] = RuntimeError("Session cancelled")
            job["event"].set()
            continue

        try:
            audio, sample_rate = synthesize_chunk(
                model=MODEL,
                text=job["text"],
                speaker=job["speaker"],
                language=job["language"],
                instruct=job.get("instruct"),
                model_id=ACTIVE_MODEL_ID,
                temperature=job.get("temperature", DEFAULT_TEMPERATURE),
                top_p=job.get("top_p", DEFAULT_TOP_P),
                repetition_penalty=job.get("repetition_penalty", DEFAULT_REPETITION_PENALTY),
            )
            job["result"] = (audio, sample_rate)
        except Exception as exc:
            job["error"] = exc
        finally:
            job["event"].set()


def start_synthesis_worker() -> None:
    configure_tts_runtime()
    thread = threading.Thread(target=synthesis_worker, daemon=True)
    thread.start()
    request_model_reload(DEFAULT_MODEL, blocking=True)


def request_model_reload(model_id: str, blocking: bool = False) -> None:
    job = {
        "type": "reload",
        "model_id": model_id,
        "event": threading.Event(),
        "error": None,
    }
    SYNTH_QUEUE.put(job)
    if blocking:
        if not job["event"].wait(timeout=300):
            raise TimeoutError(f"Timed out loading model: {model_id}")
        if job["error"] is not None:
            raise job["error"]


def synthesize_async(
    text: str,
    speaker: str,
    language: str,
    instruct: str | None = None,
    *,
    session_id: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    timeout: float = 180.0,
) -> tuple[np.ndarray, int]:
    if session_is_cancelled(session_id):
        raise RuntimeError("Session cancelled")

    if not MODEL_READY.is_set():
        if MODEL_LOADING.is_set():
            if not MODEL_READY.wait(timeout=timeout):
                raise RuntimeError("TTS model is still loading")
        else:
            # Fail fast when nothing is loading — avoids 180s hangs after a
            # failed model switch left the worker unready.
            raise RuntimeError(
                "TTS model is not ready. Select a model or restart the server."
            )
    elif MODEL_LOADING.is_set():
        raise RuntimeError("TTS model is switching. Try again in a moment.")

    if session_is_cancelled(session_id):
        raise RuntimeError("Session cancelled")

    job = {
        "text": text,
        "speaker": speaker,
        "language": language,
        "instruct": instruct,
        "session_id": session_id,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "event": threading.Event(),
        "result": None,
        "error": None,
    }
    SYNTH_QUEUE.put(job)
    if not job["event"].wait(timeout=timeout):
        raise RuntimeError("Speech synthesis timed out")
    if job["error"] is not None:
        raise job["error"]
    return job["result"]


def session_ttl_cleanup_locked() -> None:
    """Drop expired sessions. Caller must already hold SESSION_LOCK."""
    cutoff = time.time() - 3600
    expired = [key for key, value in SESSIONS.items() if value.created_at < cutoff]
    for key in expired:
        SESSIONS.pop(key, None)


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


@app.get("/")
def index():
    return render_template("index.html", build=get_build_info())


@app.get("/api/status")
def status():
    payload = get_backend_info(ACTIVE_MODEL_ID)
    payload["ready"] = MODEL_READY.is_set() and not MODEL_LOADING.is_set()
    payload["loading"] = MODEL_LOADING.is_set()
    payload["build"] = get_build_info()
    return jsonify(payload)


@app.get("/api/models")
def api_models():
    return jsonify({"models": list_models(ACTIVE_MODEL_ID)})


@app.post("/api/models/select")
def api_select_model():
    payload = request.get_json(silent=True) or {}
    model_id = (payload.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400

    status = model_download_status(model_id)
    if status != "ready":
        return jsonify({"error": "Download this model before selecting it."}), 400

    catalog = catalog_entry(model_id)
    if catalog and catalog.get("voice_type") not in READ_ALOUD_VOICE_TYPES:
        return jsonify(
            {"error": "This model type is not supported for read-aloud yet. Choose a preset-voice MLX model."}
        ), 400

    try:
        request_model_reload(model_id, blocking=True)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(get_backend_info(ACTIVE_MODEL_ID))


@app.post("/api/models/download")
def api_download_model():
    payload = request.get_json(silent=True) or {}
    model_id = (payload.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400

    status = model_download_status(model_id)
    if status == "ready":
        return jsonify({"status": "ready", "message": "Already downloaded"})

    def run_download() -> None:
        try:
            download_model(model_id)
        except Exception as exc:
            print(f"Download failed for {model_id}: {exc}")

    threading.Thread(target=run_download, daemon=True).start()
    return jsonify({"status": "downloading", "message": "Download started"})


@app.get("/api/voices")
def voices():
    return jsonify({"voices": list_voices(ACTIVE_MODEL_ID)})


@app.post("/api/fetch")
def api_fetch():
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        result = fetch_url(url)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Fetch failed: {exc}"}), 500


@app.post("/api/prepare")
def api_prepare():
    payload = request.get_json(silent=True) or {}
    text = clean_text(payload.get("text") or "", preserve_paragraphs=True)
    if not text:
        return jsonify({"error": "Text is required"}), 400

    model_info = catalog_entry(ACTIVE_MODEL_ID) or {}
    speaker = resolve_speaker(payload.get("speaker"), ACTIVE_MODEL_ID)
    chunk_chars = int(payload.get("chunk_chars") or 600)
    start_offset = max(0, int(payload.get("start_offset") or 0))
    start_offset = min(start_offset, len(text))
    steady_reading = payload.get("steady_reading", True)
    if isinstance(steady_reading, str):
        steady_reading = steady_reading.lower() not in {"0", "false", "no", "off"}
    if not model_info.get("supports_steady_reading"):
        steady_reading = False
    instruct = STEADY_READING_INSTRUCT if steady_reading else None
    temperature = STEADY_READING_TEMPERATURE if steady_reading else DEFAULT_TEMPERATURE
    top_p = STEADY_READING_TOP_P if steady_reading else DEFAULT_TOP_P
    chunks = chunk_text(text, chunk_chars)
    starts = chunk_offsets(chunks, text)
    start_chunk, start_ratio = playback_position_from_offset(start_offset, starts, chunks)

    session_id = str(uuid.uuid4())
    session = ReadSession(
        chunks=chunks,
        chunk_starts=starts,
        full_text=text,
        speaker=speaker,
        instruct=instruct,
        temperature=temperature,
        top_p=top_p,
    )

    with SESSION_LOCK:
        session_ttl_cleanup_locked()
        SESSIONS[session_id] = session

    return jsonify(
        {
            "session_id": session_id,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "chunk_starts": starts,
            "full_text": text,
            "start_chunk": start_chunk,
            "start_ratio": start_ratio,
        }
    )


@app.post("/api/session/<session_id>/stop")
def api_stop_session(session_id: str):
    if not cancel_session(session_id):
        return jsonify({"error": "Session not found or expired"}), 404
    return jsonify({"ok": True})


@app.get("/api/session/<session_id>/status")
def api_session_status(session_id: str):
    with SESSION_LOCK:
        session = SESSIONS.get(session_id)

    if session is None:
        return jsonify({"error": "Session not found or expired"}), 404

    return jsonify(
        {
            "chunk_count": len(session.chunks),
            "cached_chunks": sorted(session.audio_cache.keys()),
            "synthesizing_chunk": session.synthesizing_index,
            "cancelled": session.cancelled,
        }
    )


@app.get("/api/chunk/<session_id>/<int:chunk_index>")
def api_chunk(session_id: str, chunk_index: int):
    with SESSION_LOCK:
        session = SESSIONS.get(session_id)
        if session is None:
            return jsonify({"error": "Session not found or expired"}), 404
        if session.cancelled:
            return jsonify({"error": "Session cancelled"}), 499
        if chunk_index < 0 or chunk_index >= len(session.chunks):
            return jsonify({"error": "Chunk not found"}), 404

        cached = session.audio_cache.get(chunk_index)
        if cached is not None:
            return send_file(
                io.BytesIO(cached),
                mimetype="audio/wav",
                download_name=f"chunk_{chunk_index + 1}.wav",
            )

        session.synthesizing_index = chunk_index
        chunk_text = session.chunks[chunk_index]
        speaker = session.speaker
        language = session.language
        instruct = session.instruct
        temperature = session.temperature
        top_p = session.top_p
        repetition_penalty = session.repetition_penalty
        total_chunks = len(session.chunks)

    try:
        audio, sample_rate = synthesize_async(
            text=chunk_text,
            speaker=speaker,
            language=language,
            instruct=steady_chunk_instruct(
                instruct,
                chunk_index,
                total_chunks,
            ),
            session_id=session_id,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        with SESSION_LOCK:
            if session.synthesizing_index == chunk_index:
                session.synthesizing_index = None

    wav = wav_bytes(audio, sample_rate)
    with SESSION_LOCK:
        # Re-check cancellation; another request may have filled the cache.
        if session.cancelled:
            return jsonify({"error": "Session cancelled"}), 499
        existing = session.audio_cache.get(chunk_index)
        if existing is not None:
            wav = existing
        else:
            session.audio_cache[chunk_index] = wav

    return send_file(
        io.BytesIO(wav),
        mimetype="audio/wav",
        download_name=f"chunk_{chunk_index + 1}.wav",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Read Aloud web GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--url", help="Optional URL to pre-load in the UI")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    start_synthesis_worker()

    launch_url = f"http://{args.host}:{args.port}/"
    if args.url:
        launch_url = f"{launch_url}?url={requests_quote(args.url)}"

    print(f"Read Aloud GUI: {launch_url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(launch_url)).start()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


def requests_quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


if __name__ == "__main__":
    raise SystemExit(main())