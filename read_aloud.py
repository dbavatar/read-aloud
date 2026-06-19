#!/usr/bin/env python3
"""Read large blocks of text aloud using local Qwen3-TTS."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal

from tts_engine import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    SPEAKERS,
    get_backend_info,
    load_model,
    synthesize_chunk,
)

DEFAULT_CHUNK_CHARS = 400


def load_text(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return input("Enter text to read aloud: ").strip()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_#>]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(sentence) <= max_chars:
            current = sentence
            continue

        words = sentence.split()
        part = ""
        for word in words:
            candidate = f"{part} {word}".strip() if part else word
            if len(candidate) <= max_chars:
                part = candidate
            else:
                if part:
                    chunks.append(part)
                part = word
        if part:
            current = part

    if current:
        chunks.append(current)

    return chunks


def adjust_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0:
        return audio

    new_length = max(1, int(round(len(audio) / speed)))
    return signal.resample(audio, new_length).astype(np.float32)


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    sd.play(audio, sample_rate)
    sd.wait()


def append_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing, sr = sf.read(path, dtype="float32")
        if sr != sample_rate:
            raise ValueError(f"Sample rate mismatch for {path}: {sr} vs {sample_rate}")
        audio = np.concatenate([existing, audio])
    sf.write(path, audio, sample_rate)


def list_speakers() -> None:
    print("Available speakers:")
    for key, name in SPEAKERS.items():
        marker = "*" if key == "ryan" else " "
        print(f"  {marker} {key:10} -> {name}")
    print("\nDefault: ryan (English, clear male voice)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read large blocks of text aloud with local Qwen3-TTS."
    )
    parser.add_argument("--text", help="Text to read aloud")
    parser.add_argument("--file", "-f", help="Text file to read aloud")
    parser.add_argument(
        "--speaker",
        default=DEFAULT_SPEAKER.lower(),
        help=f"Speaker voice (default: {DEFAULT_SPEAKER.lower()})",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--instruct",
        default="",
        help="Optional style instruction, e.g. 'Read calmly and clearly'",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Qwen3-TTS model id or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help=f"Max characters per synthesis chunk (default: {DEFAULT_CHUNK_CHARS})",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional WAV file to save the full reading",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier, e.g. 2.0 for 2x (default: 1.0)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Generate audio without playing it live",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List available speaker voices",
    )
    args = parser.parse_args()

    if args.list_speakers:
        list_speakers()
        return 0

    if args.speed <= 0:
        print("Speed must be greater than 0.", file=sys.stderr)
        return 1

    text = clean_text(load_text(args))
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1

    speaker_key = args.speaker.lower().replace("-", "_")
    speaker = SPEAKERS.get(speaker_key, args.speaker)
    if speaker_key not in SPEAKERS:
        print(f"Using custom speaker name: {speaker}")

    chunks = chunk_text(text, args.chunk_chars)
    print(f"Prepared {len(chunks)} chunk(s), {len(text)} characters total.")
    if args.speed != 1.0:
        print(f"Playback speed: {args.speed}x")

    model = load_model(args.model)
    instruct = args.instruct.strip() or None
    output_path = Path(args.output) if args.output else None
    if output_path and output_path.exists():
        output_path.unlink()

    started = time.time()
    for index, chunk in enumerate(chunks, start=1):
        preview = chunk if len(chunk) <= 80 else f"{chunk[:77]}..."
        print(f"[{index}/{len(chunks)}] {preview}")

        audio, sample_rate = synthesize_chunk(
            model=model,
            text=chunk,
            speaker=speaker,
            language=args.language,
            instruct=instruct,
        )

        audio = adjust_speed(audio, args.speed)

        if output_path:
            append_wav(output_path, audio, sample_rate)

        if not args.no_play:
            play_audio(audio, sample_rate)

    elapsed = time.time() - started
    if output_path:
        print(f"Saved audio to {output_path}")
    print(f"Finished in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())