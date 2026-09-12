#!/usr/bin/env python3
"""Local web GUI for reading webpages and pasted text aloud."""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
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
    INIT_JOB_ID,
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
    model_download_status,
    configure_tts_runtime,
    resolve_speaker,
    set_current_model,
    steady_chunk_instruct,
    tts_process_main,
)
from build_info import get_build_info
from url_fetcher import fetch_url

app = Flask(__name__)

MODEL_READY = threading.Event()
MODEL_LOADING = threading.Event()
ACTIVE_MODEL_ID = DEFAULT_MODEL
SESSIONS: dict[str, "ReadSession"] = {}
SESSION_LOCK = threading.Lock()
SYNTH_TIMEOUT = 180.0
MODEL_LOAD_TIMEOUT = 300.0
MP_CTX = mp.get_context("spawn")

_WORKER_LOCK = threading.Lock()
_WORKER: "_WorkerHandle | None" = None
_PENDING_LOCK = threading.Lock()
_PENDING: dict[str, dict[str, Any]] = {}
_DISPATCHER_STARTED = threading.Event()
_WORKER_INIT_ERROR: str | None = None
_WORKER_MEMORY: dict[str, float | None] = {"active_gb": None, "peak_gb": None}


@dataclass
class _WorkerHandle:
    process: mp.Process
    job_queue: Any
    result_queue: Any
    started_at: float


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
    inflight: dict[int, threading.Event] = field(default_factory=dict)
    inflight_error: dict[int, str] = field(default_factory=dict)
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


def _fail_pending(reason: str, *, keep_init: bool = False) -> None:
    with _PENDING_LOCK:
        items = list(_PENDING.items())
        if not keep_init:
            _PENDING.clear()
        else:
            for key, _value in items:
                if key != INIT_JOB_ID:
                    _PENDING.pop(key, None)
    for job_id, rec in items:
        if keep_init and job_id == INIT_JOB_ID:
            continue
        rec["result"] = {"ok": False, "error": reason}
        rec["event"].set()


def _register_pending(job_id: str) -> dict[str, Any]:
    rec = {"event": threading.Event(), "result": None}
    with _PENDING_LOCK:
        _PENDING[job_id] = rec
    return rec


def _pop_pending(job_id: str) -> dict[str, Any] | None:
    with _PENDING_LOCK:
        return _PENDING.pop(job_id, None)


def _result_dispatcher() -> None:
    while True:
        handle = _WORKER
        if handle is None:
            time.sleep(0.05)
            continue
        try:
            result = handle.result_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        except (OSError, EOFError, ValueError, BrokenPipeError):
            time.sleep(0.1)
            continue
        if not isinstance(result, dict):
            continue
        rec = _pop_pending(str(result.get("id")))
        if rec is None:
            continue
        rec["result"] = result
        memory = result.get("memory")
        if isinstance(memory, dict):
            _WORKER_MEMORY.update(memory)
        rec["event"].set()


def _worker_monitor() -> None:
    while True:
        time.sleep(1.0)
        handle = _WORKER
        if handle is None or handle.process.is_alive():
            continue
        if MODEL_LOADING.is_set() and time.time() - handle.started_at < 5:
            continue
        print("TTS worker process died; restarting", flush=True)
        try:
            restart_worker("TTS worker process died")
        except Exception as exc:
            print(f"Failed to restart TTS worker: {exc}", flush=True)


def _start_dispatcher_once() -> None:
    if _DISPATCHER_STARTED.is_set():
        return
    threading.Thread(target=_result_dispatcher, name="tts-dispatcher", daemon=True).start()
    threading.Thread(target=_worker_monitor, name="tts-monitor", daemon=True).start()
    _DISPATCHER_STARTED.set()


def _spawn_worker_locked() -> tuple["_WorkerHandle", dict[str, Any]]:
    global _WORKER, _WORKER_INIT_ERROR, ACTIVE_MODEL_ID

    job_queue = MP_CTX.Queue()
    result_queue = MP_CTX.Queue()
    process = MP_CTX.Process(
        target=tts_process_main,
        args=(job_queue, result_queue, ACTIVE_MODEL_ID),
        daemon=True,
        name="tts-worker",
    )
    handle = _WorkerHandle(
        process=process,
        job_queue=job_queue,
        result_queue=result_queue,
        started_at=time.time(),
    )
    _WORKER_INIT_ERROR = None
    MODEL_READY.clear()
    MODEL_LOADING.set()
    rec = _register_pending(INIT_JOB_ID)
    process.start()
    _WORKER = handle
    return handle, rec


def _await_worker_ready(rec: dict[str, Any], timeout: float = MODEL_LOAD_TIMEOUT) -> None:
    global _WORKER_INIT_ERROR, ACTIVE_MODEL_ID

    if not rec["event"].wait(timeout=timeout):
        _pop_pending(INIT_JOB_ID)
        raise TimeoutError("Timed out loading TTS model")
    result = rec["result"] or {}
    if not result.get("ok"):
        _WORKER_INIT_ERROR = str(result.get("error") or "TTS model failed to load")
        MODEL_LOADING.clear()
        raise RuntimeError(_WORKER_INIT_ERROR)
    model_id = result.get("model_id") or ACTIVE_MODEL_ID
    ACTIVE_MODEL_ID = model_id
    set_current_model(model_id)
    MODEL_LOADING.clear()
    MODEL_READY.set()


def _terminate_handle(handle: _WorkerHandle | None) -> None:
    if handle is None:
        return
    process = handle.process
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)


def restart_worker(reason: str) -> None:
    print(f"Restarting TTS worker: {reason}", flush=True)
    with _WORKER_LOCK:
        handle = _WORKER
        recently_started = handle is not None and time.time() - handle.started_at < 8
        if recently_started and handle.process.is_alive() and MODEL_LOADING.is_set():
            return
        _fail_pending(reason, keep_init=False)
        _terminate_handle(handle)
        _handle, rec = _spawn_worker_locked()
    def _ready() -> None:
        try:
            _await_worker_ready(rec)
        except Exception as exc:
            print(f"TTS worker failed to become ready: {exc}", flush=True)

    threading.Thread(target=_ready, name="tts-reload", daemon=True).start()


def start_synthesis_worker() -> None:
    configure_tts_runtime()
    _start_dispatcher_once()
    with _WORKER_LOCK:
        _handle, rec = _spawn_worker_locked()
    _await_worker_ready(rec)


def request_model_reload(model_id: str, blocking: bool = False) -> None:
    global ACTIVE_MODEL_ID

    if not MODEL_READY.is_set():
        if MODEL_LOADING.is_set():
            if not MODEL_READY.wait(timeout=MODEL_LOAD_TIMEOUT):
                raise RuntimeError("TTS model is still loading")
        else:
            raise RuntimeError("TTS model is not ready. Select a model or restart the server.")

    job_id = str(uuid.uuid4())
    rec = _register_pending(job_id)
    MODEL_READY.clear()
    MODEL_LOADING.set()
    handle = _WORKER
    if handle is None or not handle.process.is_alive():
        _pop_pending(job_id)
        rec["event"].set()
        restart_worker("TTS worker missing during model reload")
        if blocking:
            if not MODEL_READY.wait(timeout=MODEL_LOAD_TIMEOUT):
                raise TimeoutError(f"Timed out loading model: {model_id}")
            if ACTIVE_MODEL_ID != model_id:
                request_model_reload(model_id, blocking=True)
        return

    handle.job_queue.put({"id": job_id, "type": "reload", "model_id": model_id})
    if not blocking:
        def _finish() -> None:
            global ACTIVE_MODEL_ID
            try:
                if not rec["event"].wait(timeout=MODEL_LOAD_TIMEOUT):
                    restart_worker(f"Timed out loading model: {model_id}")
                    return
                result = rec["result"] or {}
                if result.get("ok"):
                    loaded = result.get("model_id") or model_id
                    ACTIVE_MODEL_ID = loaded
                    set_current_model(loaded)
                MODEL_LOADING.clear()
                MODEL_READY.set()
            except Exception as exc:
                print(f"Model reload waiter failed: {exc}", flush=True)

        threading.Thread(target=_finish, name="tts-reload-wait", daemon=True).start()
        return

    if not rec["event"].wait(timeout=MODEL_LOAD_TIMEOUT):
        restart_worker(f"Timed out loading model: {model_id}")
        raise TimeoutError(f"Timed out loading model: {model_id}")
    result = rec["result"] or {}
    MODEL_LOADING.clear()
    MODEL_READY.set()
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error") or f"Failed to load {model_id}"))
    loaded = result.get("model_id") or model_id
    ACTIVE_MODEL_ID = loaded
    set_current_model(loaded)


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
    timeout: float = SYNTH_TIMEOUT,
) -> tuple[np.ndarray, int]:
    if session_is_cancelled(session_id):
        raise RuntimeError("Session cancelled")

    if not MODEL_READY.is_set():
        if MODEL_LOADING.is_set():
            if not MODEL_READY.wait(timeout=timeout):
                raise RuntimeError("TTS model is still loading")
        else:
            raise RuntimeError(
                "TTS model is not ready. Select a model or restart the server."
            )
    elif MODEL_LOADING.is_set():
        raise RuntimeError("TTS model is switching. Try again in a moment.")

    if session_is_cancelled(session_id):
        raise RuntimeError("Session cancelled")

    handle = _WORKER
    if handle is None or not handle.process.is_alive():
        restart_worker("TTS worker is not running")
        raise RuntimeError("TTS engine restarted. Press Play again.")

    job_id = str(uuid.uuid4())
    rec = _register_pending(job_id)
    handle.job_queue.put(
        {
            "id": job_id,
            "type": "synth",
            "text": text,
            "speaker": speaker,
            "language": language,
            "instruct": instruct,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
    )
    if not rec["event"].wait(timeout=timeout):
        still_waiting = False
        with _PENDING_LOCK:
            still_waiting = _PENDING.get(job_id) is rec and rec["result"] is None
        if still_waiting:
            restart_worker("Speech synthesis timed out")
            raise RuntimeError(
                "Speech synthesis timed out. TTS engine restarted; press Play again."
            )
        rec["event"].wait(timeout=1)
    result = rec["result"] or {}
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error") or "Speech synthesis failed"))
    audio = result.get("audio")
    sample_rate = result.get("sr")
    if audio is None or sample_rate is None:
        raise RuntimeError("Speech synthesis returned no audio")
    return audio, int(sample_rate)


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


def wav_response(wav: bytes, chunk_index: int):
    return send_file(
        io.BytesIO(wav),
        mimetype="audio/wav",
        download_name=f"chunk_{chunk_index + 1}.wav",
    )


@app.get("/")
def index():
    return render_template("index.html", build=get_build_info())


@app.get("/api/status")
def status():
    payload = get_backend_info(ACTIVE_MODEL_ID)
    payload["ready"] = MODEL_READY.is_set() and not MODEL_LOADING.is_set()
    payload["loading"] = MODEL_LOADING.is_set()
    payload["build"] = get_build_info()
    if _WORKER_MEMORY.get("active_gb") is not None:
        payload["memory_active_gb"] = _WORKER_MEMORY.get("active_gb")
        payload["memory_peak_gb"] = _WORKER_MEMORY.get("peak_gb")
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
    owner = False
    wait_event: threading.Event | None = None
    cached: bytes | None = None
    chunk_text = ""
    speaker = ""
    language = "English"
    instruct: str | None = None
    temperature = DEFAULT_TEMPERATURE
    top_p = DEFAULT_TOP_P
    repetition_penalty = DEFAULT_REPETITION_PENALTY
    total_chunks = 0

    with SESSION_LOCK:
        session = SESSIONS.get(session_id)
        if session is None:
            return jsonify({"error": "Session not found or expired"}), 404
        if session.cancelled:
            return jsonify({"error": "Session cancelled"}), 499
        if chunk_index < 0 or chunk_index >= len(session.chunks):
            return jsonify({"error": "Chunk not found"}), 404

        cached = session.audio_cache.get(chunk_index)
        if cached is None:
            wait_event = session.inflight.get(chunk_index)
            if wait_event is None:
                wait_event = threading.Event()
                session.inflight[chunk_index] = wait_event
                session.inflight_error.pop(chunk_index, None)
                session.synthesizing_index = chunk_index
                owner = True
                chunk_text = session.chunks[chunk_index]
                speaker = session.speaker
                language = session.language
                instruct = session.instruct
                temperature = session.temperature
                top_p = session.top_p
                repetition_penalty = session.repetition_penalty
                total_chunks = len(session.chunks)

    if cached is not None:
        return wav_response(cached, chunk_index)

    assert wait_event is not None
    if not owner:
        if not wait_event.wait(timeout=SYNTH_TIMEOUT + 15):
            return jsonify({"error": "Speech synthesis timed out"}), 500
        with SESSION_LOCK:
            cached = session.audio_cache.get(chunk_index)
            err = session.inflight_error.get(chunk_index)
            cancelled = session.cancelled
        if cancelled:
            return jsonify({"error": "Session cancelled"}), 499
        if cached is not None:
            return wav_response(cached, chunk_index)
        return jsonify({"error": err or "Speech synthesis failed"}), 500

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
        wav = wav_bytes(audio, sample_rate)
        with SESSION_LOCK:
            if session.cancelled:
                return jsonify({"error": "Session cancelled"}), 499
            session.audio_cache.setdefault(chunk_index, wav)
            wav = session.audio_cache[chunk_index]
        return wav_response(wav, chunk_index)
    except Exception as exc:
        with SESSION_LOCK:
            session.inflight_error[chunk_index] = str(exc)
        return jsonify({"error": str(exc)}), 500
    finally:
        with SESSION_LOCK:
            session.inflight.pop(chunk_index, None)
            if session.synthesizing_index == chunk_index:
                session.synthesizing_index = None
        wait_event.set()


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