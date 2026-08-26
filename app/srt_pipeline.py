from __future__ import annotations

import json
import math
import mimetypes
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


@dataclass
class Segment:
    start: float
    end: float
    text: str


PROMPT = """
You are a precise bilingual Khmer/English subtitle transcription engine.

Return ONLY valid JSON in this exact shape:
{
  "segments": [
    {"start": 0.00, "end": 2.35, "text": "ខ្មែរ..."}
  ]
}

Task:
- Transcribe exactly what is spoken.
- If the speaker says Khmer, write Khmer script.
- If the speaker says English, write English words in Latin script.
- If the speaker mixes Khmer and English, preserve that mix exactly.
- Do not translate English words into Khmer and do not romanize Khmer words.
- Segment by spoken word or very short phrase, not full sentences.
- Put spaces between spoken units so each word/phrase can become its own subtitle beat.
- Example text split style: "ជួយ មាន អត់ មាន ឃើញ" instead of "ជួយមានអត់មានឃើញ".
- Use precise start/end times in seconds.
- Each segment should contain 1 to 3 spoken units maximum.
- Do not summarize. Do not add explanations. Do not invent speech.
- If speech is unclear, write the best same-language guess and keep timing.
""".strip()


def generate_srt(input_path: Path, output_dir: Path, api_key: str, model: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    media_path = _prepare_media(input_path, output_dir)
    duration = _duration_seconds(media_path)
    segments = _transcribe_with_gemini(media_path, api_key, model)
    beat_segments = _split_into_subtitle_beats(segments)
    corrected = _correct_timing(beat_segments, duration)
    srt_path = output_dir / f"{input_path.stem}.km.srt"
    srt_path.write_text(_to_srt(corrected), encoding="utf-8")
    return srt_path


def _prepare_media(input_path: Path, output_dir: Path) -> Path:
    if input_path.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}:
        return input_path

    audio_path = output_dir / f"{input_path.stem}.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return audio_path


def _duration_seconds(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _transcribe_with_gemini(path: Path, api_key: str, model: str) -> list[Segment]:
    client = genai.Client(api_key=api_key)
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    media_bytes = path.read_bytes()

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=media_bytes, mime_type=mime_type),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    payload = _load_json(response.text or "")
    raw_segments = payload.get("segments", [])
    return [_segment_from_json(item) for item in raw_segments]


def _load_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _segment_from_json(item: dict[str, Any]) -> Segment:
    text = str(item.get("text", "")).strip()
    start = float(item.get("start", 0))
    end = float(item.get("end", start + 1.5))
    return Segment(start=max(0.0, start), end=max(0.0, end), text=text)


def _correct_timing(segments: list[Segment], duration: float | None) -> list[Segment]:
    usable = [s for s in segments if s.text]
    usable.sort(key=lambda s: (s.start, s.end))
    corrected: list[Segment] = []

    for segment in usable:
        start = segment.start
        end = segment.end

        if corrected:
            previous = corrected[-1]
            min_gap = 0.06
            if start < previous.end + min_gap:
                start = previous.end + min_gap

        min_duration = min(0.75, max(0.28, len(segment.text) / 28.0))
        max_duration = max(0.7, min(1.35, len(segment.text) / 8.5))
        if end <= start:
            end = start + min_duration
        end = max(end, start + min_duration)
        end = min(end, start + max_duration)

        if duration is not None:
            start = min(start, max(0.0, duration - 0.1))
            end = min(end, duration)

        if end > start:
            corrected.append(Segment(start=start, end=end, text=_clean_text(segment.text)))

    return corrected


def _split_into_subtitle_beats(segments: list[Segment]) -> list[Segment]:
    beats: list[Segment] = []
    for segment in segments:
        tokens = _tokenize_spoken_units(segment.text)
        if not tokens:
            continue
        if len(tokens) == 1:
            beats.append(Segment(segment.start, segment.end, tokens[0]))
            continue

        total_weight = sum(_token_weight(token) for token in tokens)
        duration = max(0.01, segment.end - segment.start)
        cursor = segment.start
        for index, token in enumerate(tokens):
            if index == len(tokens) - 1:
                end = segment.end
            else:
                end = cursor + duration * (_token_weight(token) / total_weight)
            beats.append(Segment(cursor, end, token))
            cursor = end
    return beats


def _tokenize_spoken_units(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    spaced = re.sub(r"([,។៕!?;:])", r" \1 ", cleaned)
    raw_tokens = [token.strip(" \t\r\n,។៕!?;:") for token in spaced.split()]
    return [token for token in raw_tokens if token]


def _token_weight(token: str) -> float:
    latin_chars = len(re.findall(r"[A-Za-z0-9]", token))
    khmer_chars = len(re.findall(r"[\u1780-\u17FF]", token))
    other_chars = max(1, len(token) - latin_chars - khmer_chars)
    return max(1.0, latin_chars / 4.0 + khmer_chars / 3.0 + other_chars / 4.0)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _to_srt(segments: list[Segment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n{_fmt_time(segment.start)} --> {_fmt_time(segment.end)}\n{segment.text}"
        )
    return "\n\n".join(blocks) + "\n"


def _fmt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    millis = milliseconds % 1000
    if math.isclose(seconds, 0.0):
        hours = minutes = secs = millis = 0
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
