"""Local TTS engine for Apple Silicon via MLX."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import numpy as np
from mlx_audio.tts.utils import load_model as load_mlx_model

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
DEFAULT_SPEAKER = "Ryan"
DEFAULT_LANGUAGE = "English"
SAMPLE_RATE = 24000
STEADY_READING_INSTRUCT = (
    "Speak in a calm, neutral audiobook narrator voice. "
    "Keep tone, pace, pitch, and emotional energy identical across every sentence. "
    "Do not add excitement, sadness, anger, or emphasis unless the text is a direct quote. "
    "Read punctuation plainly without dramatic inflection."
)
STEADY_READING_TEMPERATURE = 0.5
STEADY_READING_TOP_P = 0.85
DEFAULT_TEMPERATURE = 0.65
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.05

HF_CACHE = Path.home() / ".cache/huggingface/hub"
LM_STUDIO_MLX = Path.home() / ".lmstudio/models/mlx-community"

READ_ALOUD_VOICE_TYPES = frozenset({"custom", "preset", "single"})

MODEL_CATALOG = [
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "name": "Qwen3 CustomVoice 1.7B 8-bit",
        "backend": "mlx",
        "family": "qwen",
        "size_label": "~2.4 GB",
        "description": "Recommended. Preset voices with emotion control.",
        "recommended": True,
        "voice_type": "custom",
        "supports_steady_reading": True,
    },
    {
        "id": "mlx-community/Kokoro-82M-8bit",
        "name": "Kokoro 82M 8-bit",
        "backend": "mlx",
        "family": "kokoro",
        "size_label": "~170 MB",
        "description": "Fastest option. Lightweight multilingual preset voices.",
        "voice_type": "preset",
        "lang_code": "a",
        "supports_steady_reading": False,
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        "name": "Qwen3 CustomVoice 0.6B",
        "backend": "mlx",
        "family": "qwen",
        "size_label": "~1.5 GB",
        "description": "Fast Qwen variant. Good for very long reading sessions.",
        "voice_type": "custom",
        "supports_steady_reading": True,
    },
    {
        "id": "mlx-community/kitten-tts-nano-0.8",
        "name": "KittenTTS Nano 0.8",
        "backend": "mlx",
        "family": "kitten",
        "size_label": "~50 MB",
        "description": "Tiny English TTS. Very fast, compact download.",
        "voice_type": "preset",
        "supports_steady_reading": False,
    },
    {
        "id": "mlx-community/kitten-tts-mini-0.8",
        "name": "KittenTTS Mini 0.8",
        "backend": "mlx",
        "family": "kitten",
        "size_label": "~120 MB",
        "description": "Compact English TTS with natural preset voices.",
        "voice_type": "preset",
        "supports_steady_reading": False,
    },
    {
        "id": "mlx-community/Soprano-1.1-80M-bf16",
        "name": "Soprano 80M",
        "backend": "mlx",
        "family": "soprano",
        "size_label": "~160 MB",
        "description": "Fast single-voice English TTS. No voice picker needed.",
        "voice_type": "single",
        "supports_steady_reading": False,
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        "name": "Qwen3 CustomVoice 1.7B bf16",
        "backend": "mlx",
        "family": "qwen",
        "size_label": "~3.5 GB",
        "description": "Higher precision MLX model. Slightly slower, may sound cleaner.",
        "voice_type": "custom",
        "supports_steady_reading": True,
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "name": "Qwen3 Base 1.7B",
        "backend": "mlx",
        "family": "qwen",
        "size_label": "~3.5 GB",
        "description": "Voice cloning from a short audio sample (not preset voices).",
        "voice_type": "clone",
        "supports_steady_reading": False,
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "name": "Qwen3 VoiceDesign 1.7B",
        "backend": "mlx",
        "family": "qwen",
        "size_label": "~3.5 GB",
        "description": "Design a voice from a text description.",
        "voice_type": "design",
        "supports_steady_reading": False,
    },
]

QWEN_VOICES = {
    "ryan": ("Ryan", "Ryan"),
    "aiden": ("Aiden", "Aiden"),
    "vivian": ("Vivian", "Vivian"),
    "serena": ("Serena", "Serena"),
    "uncle_fu": ("Uncle Fu", "Uncle_Fu"),
    "dylan": ("Dylan", "Dylan"),
    "eric": ("Eric", "Eric"),
    "ono_anna": ("Ono Anna", "Ono_Anna"),
    "sohee": ("Sohee", "Sohee"),
}

KOKORO_VOICES = {
    "af_heart": ("Heart (US female)", "af_heart"),
    "af_bella": ("Bella (US female)", "af_bella"),
    "af_nova": ("Nova (US female)", "af_nova"),
    "af_sarah": ("Sarah (US female)", "af_sarah"),
    "af_nicole": ("Nicole (US female)", "af_nicole"),
    "am_michael": ("Michael (US male)", "am_michael"),
    "am_fenrir": ("Fenrir (US male)", "am_fenrir"),
    "am_puck": ("Puck (US male)", "am_puck"),
    "bf_emma": ("Emma (UK female)", "bf_emma"),
    "bm_george": ("George (UK male)", "bm_george"),
}

KITTEN_VOICES = {
    "bella": ("Bella", "Bella"),
    "jasper": ("Jasper", "Jasper"),
    "luna": ("Luna", "Luna"),
    "bruno": ("Bruno", "Bruno"),
    "rosie": ("Rosie", "Rosie"),
    "hugo": ("Hugo", "Hugo"),
    "kiki": ("Kiki", "Kiki"),
    "leo": ("Leo", "Leo"),
}

# Backward-compatible alias used by read_aloud_server imports.
SPEAKERS = {key: api_name for key, (_label, api_name) in QWEN_VOICES.items()}

DOWNLOAD_STATE: dict[str, dict[str, str]] = {}
DOWNLOAD_LOCK = threading.Lock()
CURRENT_MODEL_ID = DEFAULT_MODEL
_RUNTIME_CONFIGURED = False


def configure_tts_runtime() -> None:
    """Configure native/runtime deps (espeak-ng for KittenTTS, etc.)."""
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return

    if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
        library_candidates: list[Path] = []
        for prefix in ("/opt/homebrew", "/usr/local"):
            cellar = Path(prefix) / "Cellar/espeak-ng"
            if cellar.is_dir():
                library_candidates.extend(sorted(cellar.glob("*/lib/libespeak-ng.dylib"), reverse=True))
            library_candidates.append(Path(prefix) / "lib/libespeak-ng.dylib")

        for candidate in library_candidates:
            if candidate.is_file():
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(candidate.resolve())
                break

    _RUNTIME_CONFIGURED = True


def catalog_entry(model_id: str | None = None) -> dict | None:
    model_id = model_id or CURRENT_MODEL_ID
    return next((item for item in MODEL_CATALOG if item["id"] == model_id), None)


def model_family(model_id: str | None = None) -> str:
    entry = catalog_entry(model_id)
    return entry.get("family", "qwen") if entry else "qwen"


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


def _has_weights(model_dir: Path) -> bool:
    return any(model_dir.glob("*.safetensors"))


def _model_dir_ready(model_dir: Path, family: str) -> bool:
    if not (model_dir / "config.json").is_file():
        return False
    if family == "qwen":
        return _has_speech_tokenizer(model_dir)
    if family == "kokoro":
        return _has_weights(model_dir)
    if family == "kitten":
        return (model_dir / "model.safetensors").is_file() and (model_dir / "voices.npz").is_file()
    if family == "soprano":
        return _has_weights(model_dir)
    return _has_weights(model_dir)


def model_download_status(model_id: str) -> str:
    with DOWNLOAD_LOCK:
        active = DOWNLOAD_STATE.get(model_id, {}).get("status")
        if active == "downloading":
            return "downloading"

    family = model_family(model_id)
    snapshot = _hf_snapshot_path(model_id)
    if snapshot and _model_dir_ready(snapshot, family):
        return "ready"

    lmstudio = _lmstudio_model_dir(model_id)
    if lmstudio:
        if _model_dir_ready(lmstudio, family):
            return "ready"
        return "partial"

    if snapshot and any(snapshot.iterdir()):
        return "partial"
    if lmstudio and any(lmstudio.iterdir()):
        return "partial"
    return "missing"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def model_size_bytes(model_id: str) -> int | None:
    family = model_family(model_id)
    snapshot = _hf_snapshot_path(model_id)
    if snapshot and _model_dir_ready(snapshot, family):
        return _dir_size_bytes(snapshot)

    lmstudio = _lmstudio_model_dir(model_id)
    if lmstudio and _model_dir_ready(lmstudio, family):
        return _dir_size_bytes(lmstudio)
    return None


def parse_size_label_gb(size_label: str) -> float | None:
    match = re.search(r"([\d.]+)\s*GB", size_label, flags=re.IGNORECASE)
    if match:
        return round(float(match.group(1)), 2)
    match = re.search(r"([\d.]+)\s*MB", size_label, flags=re.IGNORECASE)
    if match:
        return round(float(match.group(1)) / 1000, 2)
    return None


def format_gb(value_gb: float | None) -> str | None:
    if value_gb is None:
        return None
    if value_gb >= 10:
        return f"{value_gb:.1f} GB"
    if value_gb >= 1:
        return f"{value_gb:.2f} GB"
    return f"{value_gb:.2f} GB"


def get_memory_stats() -> dict[str, float | None]:
    try:
        import mlx.core as mx

        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
        return {
            "active_gb": round(active / 1e9, 2),
            "peak_gb": round(peak / 1e9, 2),
        }
    except Exception:
        return {"active_gb": None, "peak_gb": None}


def _voice_table(model_id: str | None = None) -> dict[str, tuple[str, str]]:
    family = model_family(model_id)
    if family == "kokoro":
        return KOKORO_VOICES
    if family == "kitten":
        return KITTEN_VOICES
    return QWEN_VOICES


def list_voices(model_id: str | None = None) -> list[dict[str, str | bool]]:
    entry = catalog_entry(model_id)
    if entry and entry.get("voice_type") == "single":
        return [
            {
                "id": "default",
                "label": "Default voice",
                "default": True,
            }
        ]

    voices = []
    default_id = None
    for key, (label, _api_name) in _voice_table(model_id).items():
        if default_id is None:
            default_id = key
        voices.append(
            {
                "id": key,
                "label": label,
                "default": key == default_id,
            }
        )
    return voices


def resolve_speaker(speaker_key: str | None, model_id: str | None = None) -> str:
    entry = catalog_entry(model_id)
    if entry and entry.get("voice_type") == "single":
        return "default"

    normalized = (speaker_key or "").strip().lower().replace("-", "_")
    table = _voice_table(model_id)
    if normalized in table:
        return table[normalized][1]

    if speaker_key and speaker_key in {api_name for (_label, api_name) in table.values()}:
        return speaker_key

    first = next(iter(table.values()))
    return first[1]


def list_models(current_model_id: str | None = None) -> list[dict]:
    current = current_model_id or CURRENT_MODEL_ID
    models = []
    for entry in MODEL_CATALOG:
        model_id = entry["id"]
        status = model_download_status(model_id)
        with DOWNLOAD_LOCK:
            download_info = DOWNLOAD_STATE.get(model_id, {})
        on_disk_bytes = model_size_bytes(model_id)
        on_disk_gb = round(on_disk_bytes / 1e9, 2) if on_disk_bytes else None
        estimated_gb = parse_size_label_gb(entry.get("size_label", ""))
        models.append(
            {
                **entry,
                "status": status,
                "selected": model_id == current,
                "download_message": download_info.get("message", ""),
                "size_gb_on_disk": on_disk_gb,
                "size_gb_estimate": estimated_gb,
            }
        )
    return models


def resolve_model_path(model_name: str | None = None) -> str:
    model_name = model_name or CURRENT_MODEL_ID
    if model_name.startswith("/") or model_name.startswith("~"):
        return str(Path(model_name).expanduser())
    if os.path.isdir(model_name):
        return model_name

    family = model_family(model_name)
    snapshot = _hf_snapshot_path(model_name)
    if snapshot and _model_dir_ready(snapshot, family):
        return str(snapshot)

    lmstudio = _lmstudio_model_dir(model_name)
    if lmstudio and _model_dir_ready(lmstudio, family):
        return str(lmstudio)

    return model_name


def get_backend_info(model_name: str | None = None) -> dict[str, str | bool | float | None]:
    model_id = model_name or CURRENT_MODEL_ID
    resolved = resolve_model_path(model_id)
    entry = catalog_entry(model_id)
    on_disk_bytes = model_size_bytes(model_id)
    on_disk_gb = round(on_disk_bytes / 1e9, 2) if on_disk_bytes else None
    estimated_gb = parse_size_label_gb(entry.get("size_label", "")) if entry else None
    memory = get_memory_stats()
    return {
        "backend": "mlx-audio",
        "backend_label": "MLX",
        "device": "apple-silicon-gpu",
        "model_id": model_id,
        "model": resolved,
        "model_name": entry["name"] if entry else model_id,
        "model_status": model_download_status(model_id),
        "model_family": entry.get("family", "qwen") if entry else "qwen",
        "supports_steady_reading": bool(entry.get("supports_steady_reading")) if entry else False,
        "size_gb_on_disk": on_disk_gb,
        "size_gb_estimate": estimated_gb,
        "memory_active_gb": memory["active_gb"],
        "memory_peak_gb": memory["peak_gb"],
        "lm_studio": "not used",
    }


def set_current_model(model_id: str) -> None:
    global CURRENT_MODEL_ID
    known = {entry["id"] for entry in MODEL_CATALOG}
    if model_id not in known:
        raise ValueError(f"Unknown model: {model_id}")
    CURRENT_MODEL_ID = model_id


def mlx_load_target(model_name: str | None = None) -> str:
    """Path or HF repo id to pass to mlx-audio's loader."""
    model_name = model_name or CURRENT_MODEL_ID
    if model_name.startswith("/") or model_name.startswith("~"):
        return str(Path(model_name).expanduser())
    if os.path.isdir(model_name):
        return model_name

    lmstudio = _lmstudio_model_dir(model_name)
    family = model_family(model_name)
    if lmstudio and _model_dir_ready(lmstudio, family):
        return str(lmstudio)

    # HF repo ids let mlx-audio detect model family; snapshot hashes do not.
    return model_name


def load_model(model_name: str | None = None):
    configure_tts_runtime()
    model_name = model_name or CURRENT_MODEL_ID
    set_current_model(model_name)
    resolved = mlx_load_target(model_name)
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


def steady_chunk_instruct(
    base: str | None,
    chunk_index: int,
    total_chunks: int,
) -> str | None:
    if not base:
        return None
    if total_chunks <= 1:
        return base
    return (
        f"{base} This is segment {chunk_index + 1} of {total_chunks} in one continuous reading. "
        "Match the delivery of prior segments exactly; do not shift emotion or energy at boundaries."
    )


def _results_to_audio(results: list) -> tuple[np.ndarray, int]:
    parts: list[np.ndarray] = []
    sample_rate = SAMPLE_RATE
    for result in results:
        audio = np.asarray(result.audio, dtype=np.float32).squeeze()
        if audio.size:
            parts.append(audio)
        sample_rate = int(getattr(result, "sample_rate", sample_rate))
    if not parts:
        raise RuntimeError("TTS model returned no audio")
    if len(parts) == 1:
        return parts[0], sample_rate
    return np.concatenate(parts), sample_rate


def _sentence_segments(text: str, max_chars: int) -> list[str]:
    """Split long text into sentence-sized pieces for safer Qwen generation.

    mlx-audio caps CustomVoice generation at roughly 6 codec tokens per text
    token. Slow/steady speech can hit that cap mid-chunk and drop the ending.
    Shorter segments each get their own budget.
    """
    text = (text or "").strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []

    parts = re.split(r"(?<=[.!?])(?:\s+|\n+)", text)
    segments: list[str] = []
    current = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            segments.append(current)
        if len(piece) <= max_chars:
            current = piece
            continue
        # Hard-wrap oversized sentences on word boundaries.
        words = piece.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    segments.append(current)
                current = word
    if current:
        segments.append(current)
    return segments or [text]


def _synthesize_qwen_segment(
    model,
    text: str,
    speaker: str,
    language: str,
    instruct: str | None,
    *,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple[np.ndarray, int]:
    kwargs = {
        "text": text,
        "speaker": speaker,
        "language": language,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        # mlx-audio still applies an internal text-length cap; a high ceiling
        # avoids the outer min(max_tokens, …) from truncating first.
        "max_tokens": 8192,
    }
    if instruct:
        kwargs["instruct"] = instruct
    return _results_to_audio(list(model.generate_custom_voice(**kwargs)))


def synthesize_chunk(
    model,
    text: str,
    speaker: str,
    language: str = DEFAULT_LANGUAGE,
    instruct: str | None = None,
    *,
    model_id: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
) -> tuple[np.ndarray, int]:
    model_id = model_id or CURRENT_MODEL_ID
    family = model_family(model_id)
    entry = catalog_entry(model_id) or {}

    if family == "qwen":
        # Keep segments short enough that slow/steady delivery fits the
        # library's per-call codec-token budget (see _sentence_segments).
        segments = _sentence_segments(text, max_chars=280)
        if len(segments) <= 1:
            return _synthesize_qwen_segment(
                model,
                text,
                speaker,
                language,
                instruct,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        parts: list[np.ndarray] = []
        sample_rate = SAMPLE_RATE
        for segment in segments:
            audio, sample_rate = _synthesize_qwen_segment(
                model,
                segment,
                speaker,
                language,
                instruct,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            if audio.size:
                parts.append(audio)
        if not parts:
            raise RuntimeError("TTS model returned no audio")
        if len(parts) == 1:
            return parts[0], sample_rate
        # Tiny silence between sub-segments avoids clicks at joins.
        gap = np.zeros(int(sample_rate * 0.05), dtype=np.float32)
        joined: list[np.ndarray] = []
        for index, part in enumerate(parts):
            joined.append(part)
            if index < len(parts) - 1:
                joined.append(gap)
        return np.concatenate(joined), sample_rate

    if family == "kokoro":
        lang_code = entry.get("lang_code", "a")
        return _results_to_audio(
            list(
                model.generate(
                    text=text,
                    voice=speaker,
                    lang_code=lang_code,
                    speed=1.0,
                )
            )
        )

    if family == "kitten":
        return _results_to_audio(
            list(
                model.generate(
                    text=text,
                    voice=speaker,
                    speed=1.0,
                )
            )
        )

    if family == "soprano":
        return _results_to_audio(list(model.generate(text=text, temperature=temperature, top_p=top_p)))

    raise RuntimeError(f"Unsupported TTS model family: {family}")


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