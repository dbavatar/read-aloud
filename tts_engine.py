"""Local TTS engine for Apple Silicon via MLX."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from mlx_audio.tts.utils import load_model as load_mlx_model

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
DEFAULT_SPEAKER = "Ryan"
DEFAULT_LANGUAGE = "English"
SAMPLE_RATE = 24000
STEADY_READING_INSTRUCT = (
    "Speak in a calm, steady audiobook narrator voice. "
    "Keep tone, pace, and emotion consistent throughout. Neutral delivery."
)
DEFAULT_TEMPERATURE = 0.65
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.05

HF_CACHE = Path.home() / ".cache/huggingface/hub"
LM_STUDIO_MLX = Path.home() / ".lmstudio/models/mlx-community"

MODEL_CATALOG = [
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "name": "Qwen3 CustomVoice 1.7B 8-bit",
        "backend": "mlx",
        "size_label": "~2.4 GB",
        "description": "Recommended. Preset voices with emotion control.",
        "recommended": True,
        "voice_type": "custom",
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        "name": "Qwen3 CustomVoice 1.7B bf16",
        "backend": "mlx",
        "size_label": "~3.5 GB",
        "description": "Higher precision MLX model. Slightly slower, may sound cleaner.",
        "voice_type": "custom",
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        "name": "Qwen3 CustomVoice 0.6B",
        "backend": "mlx",
        "size_label": "~1.5 GB",
        "description": "Fastest MLX option. Good for very long reading sessions.",
        "voice_type": "custom",
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "name": "Qwen3 Base 1.7B",
        "backend": "mlx",
        "size_label": "~3.5 GB",
        "description": "Voice cloning from a short audio sample (not preset voices).",
        "voice_type": "clone",
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "name": "Qwen3 VoiceDesign 1.7B",
        "backend": "mlx",
        "size_label": "~3.5 GB",
        "description": "Design a voice from a text description.",
        "voice_type": "design",
    },
]

SPEAKERS = {
    "ryan": "Ryan",
    "aiden": "Aiden",
    "vivian": "Vivian",
    "serena": "Serena",
    "uncle_fu": "Uncle_Fu",
    "dylan": "Dylan",
    "eric": "Eric",
    "ono_anna": "Ono_Anna",
    "sohee": "Sohee",
}

DOWNLOAD_STATE: dict[str, dict[str, str]] = {}
DOWNLOAD_LOCK = threading.Lock()
CURRENT_MODEL_ID = DEFAULT_MODEL


def _repo_cache_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _hf_snapshot_path(model_id: str) -> Path | None:
    repo_dir = HF_CACHE / _repo_cache_name(model_id)
    if not repo_dir.is_dir():
        return None
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _lmstudio_model_dir(model_id: str) -> Path | None:
    short_name = model_id.split("/", 1)[-1]
    path = LM_STUDIO_MLX / short_name
    return path if path.is_dir() else None


def _has_speech_tokenizer(model_dir: Path) -> bool:
    return (model_dir / "speech_tokenizer").is_dir()


def model_download_status(model_id: str) -> str:
    with DOWNLOAD_LOCK:
        active = DOWNLOAD_STATE.get(model_id, {}).get("status")
        if active == "downloading":
            return "downloading"

    snapshot = _hf_snapshot_path(model_id)
    if snapshot and _has_speech_tokenizer(snapshot):
        return "ready"

    lmstudio = _lmstudio_model_dir(model_id)
    if lmstudio:
        if _has_speech_tokenizer(lmstudio):
            return "ready"
        return "partial"

    if snapshot and any(snapshot.iterdir()):
        return "partial"
    if lmstudio and any(lmstudio.iterdir()):
        return "partial"
    return "missing"


def list_models(current_model_id: str | None = None) -> list[dict]:
    current = current_model_id or CURRENT_MODEL_ID
    models = []
    for entry in MODEL_CATALOG:
        model_id = entry["id"]
        status = model_download_status(model_id)
        with DOWNLOAD_LOCK:
            download_info = DOWNLOAD_STATE.get(model_id, {})
        models.append(
            {
                **entry,
                "status": status,
                "selected": model_id == current,
                "download_message": download_info.get("message", ""),
            }
        )
    return models


def resolve_model_path(model_name: str | None = None) -> str:
    model_name = model_name or CURRENT_MODEL_ID
    if model_name.startswith("/") or model_name.startswith("~"):
        return str(Path(model_name).expanduser())
    if os.path.isdir(model_name):
        return model_name

    snapshot = _hf_snapshot_path(model_name)
    if snapshot and _has_speech_tokenizer(snapshot):
        return str(snapshot)

    lmstudio = _lmstudio_model_dir(model_name)
    if lmstudio and _has_speech_tokenizer(lmstudio):
        return str(lmstudio)

    return model_name


def get_backend_info(model_name: str | None = None) -> dict[str, str | bool]:
    model_id = model_name or CURRENT_MODEL_ID
    resolved = resolve_model_path(model_id)
    catalog = next((item for item in MODEL_CATALOG if item["id"] == model_id), None)
    return {
        "backend": "mlx-audio",
        "backend_label": "MLX",
        "device": "apple-silicon-gpu",
        "model_id": model_id,
        "model": resolved,
        "model_name": catalog["name"] if catalog else model_id,
        "model_status": model_download_status(model_id),
        "lm_studio": "not used",
    }


def set_current_model(model_id: str) -> None:
    global CURRENT_MODEL_ID
    known = {entry["id"] for entry in MODEL_CATALOG}
    if model_id not in known:
        raise ValueError(f"Unknown model: {model_id}")
    CURRENT_MODEL_ID = model_id


def load_model(model_name: str | None = None):
    model_name = model_name or CURRENT_MODEL_ID
    set_current_model(model_name)
    resolved = resolve_model_path(model_name)
    print(f"Loading MLX TTS model: {resolved}")
    return load_mlx_model(resolved)


def download_model(model_id: str) -> None:
    known = {entry["id"] for entry in MODEL_CATALOG}
    if model_id not in known:
        raise ValueError(f"Unknown model: {model_id}")

    with DOWNLOAD_LOCK:
        if DOWNLOAD_STATE.get(model_id, {}).get("status") == "downloading":
            return
        DOWNLOAD_STATE[model_id] = {"status": "downloading", "message": "Starting download…"}

    try:
        from huggingface_hub import snapshot_download

        with DOWNLOAD_LOCK:
            DOWNLOAD_STATE[model_id]["message"] = "Downloading from Hugging Face…"

        snapshot_download(repo_id=model_id)
        status = model_download_status(model_id)
        message = "Download complete" if status == "ready" else "Download finished but model may be incomplete"
        with DOWNLOAD_LOCK:
            DOWNLOAD_STATE[model_id] = {"status": "idle", "message": message}
    except Exception as exc:
        with DOWNLOAD_LOCK:
            DOWNLOAD_STATE[model_id] = {"status": "error", "message": str(exc)}
        raise


def synthesize_chunk(
    model,
    text: str,
    speaker: str,
    language: str = DEFAULT_LANGUAGE,
    instruct: str | None = None,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
) -> tuple[np.ndarray, int]:
    kwargs = {
        "text": text,
        "speaker": speaker,
        "language": language,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    if instruct:
        kwargs["instruct"] = instruct

    results = list(model.generate_custom_voice(**kwargs))
    if not results:
        raise RuntimeError("TTS model returned no audio")

    result = results[0]
    audio = np.asarray(result.audio, dtype=np.float32).squeeze()
    sample_rate = int(getattr(result, "sample_rate", SAMPLE_RATE))
    return audio, sample_rate


def chunk_offsets(chunks: list[str], full_text: str) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for chunk in chunks:
        idx = full_text.find(chunk, cursor)
        if idx == -1:
            idx = cursor
        offsets.append(idx)
        cursor = idx + len(chunk)
    return offsets