from __future__ import annotations

import json
import math
import mimetypes
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    {
      "start": 0.00,
      "end": 2.35,
      "text": "ខ្ញុំស្រលាញ់បង",
      "words": [
        {"start": 0.00, "end": 0.55, "text": "ខ្ញុំ"},
        {"start": 0.55, "end": 1.60, "text": "ស្រលាញ់"},
        {"start": 1.60, "end": 2.35, "text": "បង"}
      ]
    }
  ]
}

Task:
- Accuracy is the first priority. Transcribe the exact words the speaker says.
- Do not guess a different sentence just to make subtitles short.
- First hear the full spoken phrase accurately, then split it into word timings.
- If the speaker says Khmer, write Khmer script.
- If the speaker says English, write English words in Latin script.
- If the speaker mixes Khmer and English, preserve that mix exactly.
- Do not translate English words into Khmer and do not romanize Khmer words.
- Keep segment.text as the natural full phrase that was spoken.
- In segment.words, split that exact phrase into spoken words/units with accurate start/end times.
- Khmer example: if the speaker says "ខ្ញុំស្រលាញ់បង", text must be "ខ្ញុំស្រលាញ់បង" and words must be ["ខ្ញុំ", "ស្រលាញ់", "បង"].
- Timing example: if "អាបង" is spoken for 2 seconds, the subtitle timing must cover 2 seconds. If it is spoken for 1 second, it must cover 1 second.
- Use precise start/end times in seconds.
- Do not merge multiple spoken words into one word item unless Khmer spelling truly requires it.
- Do not summarize. Do not add explanations. Do not invent speech.
- If speech is unclear, write the best same-language guess and keep timing.
""".strip()

REPAIR_PROMPT = """
Return ONLY valid JSON in this exact shape:
{"segments":[{"start":0.0,"end":1.0,"text":"spoken words","words":[{"start":0.0,"end":1.0,"text":"spoken"}]}]}

Fix this invalid subtitle JSON. Keep the same data. Remove any prose, markdown, or trailing broken text.
""".strip()

MODEL_FALLBACKS = {
    "gemini-3.6-flash": ("gemini-2.5-flash", "gemini-2.0-flash"),
    "gemini-2.5-pro": ("gemini-2.5-flash", "gemini-2.0-flash"),
}


def generate_srt(
    input_path: Path,
    output_dir: Path,
    api_key: str,
    model: str,
    words_per_subtitle: int = 1,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    media_path = _prepare_media(input_path, output_dir)
    duration = _duration_seconds(media_path)
    segments = _transcribe_with_gemini(media_path, api_key, model)
    beat_segments = _split_into_subtitle_beats(segments, words_per_subtitle)
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
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    media_bytes = path.read_bytes()
    models_to_try = [model, *MODEL_FALLBACKS.get(model, ())]
    last_error: Exception | None = None

    for candidate_model in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=[
                        types.Part.from_bytes(data=media_bytes, mime_type=mime_type),
                        PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                payload = _load_json(response.text or "")
                return _segments_from_payload(payload)
            except json.JSONDecodeError as exc:
                last_error = exc
                repaired = _repair_json_with_gemini(client, types, candidate_model, response.text or "")
                if repaired:
                    return _segments_from_payload(repaired)
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc):
                    break
                time.sleep(2**attempt)

    raise RuntimeError(f"Gemini transcription failed after retries: {last_error}")


def _repair_json_with_gemini(client: Any, types: Any, model: str, bad_text: str) -> dict[str, Any] | None:
    candidate = _extract_json_object(bad_text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    try:
        response = client.models.generate_content(
            model=model,
            contents=[REPAIR_PROMPT, bad_text[:12000]],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return _load_json(response.text or "")
    except Exception:
        return None


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "503",
            "unavailable",
            "timed out",
            "timeout",
            "deadline",
            "rate limit",
            "temporarily",
        )
    )


def _load_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = _extract_json_object(cleaned)
        if extracted:
            return json.loads(extracted)
        raise


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _segment_from_json(item: dict[str, Any]) -> Segment:
    text = str(item.get("text", "")).strip()
    start = float(item.get("start", 0))
    end = float(item.get("end", start + 1.5))
    return Segment(start=max(0.0, start), end=max(0.0, end), text=text)


def _segments_from_payload(payload: dict[str, Any]) -> list[Segment]:
    segments: list[Segment] = []
    for item in payload.get("segments", []):
        word_items = item.get("words")
        if isinstance(word_items, list) and word_items:
            segments.extend(_word_segments_from_json(item, word_items))
        else:
            segments.append(_segment_from_json(item))
    return segments


def _word_segments_from_json(parent: dict[str, Any], word_items: list[Any]) -> list[Segment]:
    parent_start = float(parent.get("start", 0))
    parent_end = float(parent.get("end", parent_start + 1.5))
    segments: list[Segment] = []

    for item in word_items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.get("start", parent_start))
        end = float(item.get("end", start))
        start = min(max(parent_start, start), parent_end)
        end = min(max(start, end), parent_end)
        if end <= start:
            end = min(parent_end, start + 0.12)
        segments.append(Segment(start=max(0.0, start), end=max(0.0, end), text=text))

    return segments or [_segment_from_json(parent)]


def _correct_timing(segments: list[Segment], duration: float | None) -> list[Segment]:
    usable = [s for s in segments if s.text]
    usable.sort(key=lambda s: (s.start, s.end))
    corrected: list[Segment] = []

    for segment in usable:
        start = segment.start
        end = segment.end

        if corrected:
            previous = corrected[-1]
            min_gap = 0.01
            if start < previous.end + min_gap:
                previous.end = max(previous.start + 0.08, start - min_gap)

        min_duration = min(0.35, max(0.12, len(segment.text) / 60.0))
        max_duration = max(0.32, min(0.9, len(segment.text) / 13.0))
        if end <= start:
            end = start + min_duration
        elif end - start < min_duration:
            next_start = _next_start_after(usable, segment)
            available_end = next_start - 0.01 if next_start is not None else None
            end = start + min_duration
            if available_end is not None:
                end = min(end, available_end)
        end = min(end, start + max_duration)

        if duration is not None:
            start = min(start, max(0.0, duration - 0.1))
            end = min(end, duration)

        if end > start:
            corrected.append(Segment(start=start, end=end, text=_clean_text(segment.text)))

    return corrected


def _next_start_after(segments: list[Segment], current: Segment) -> float | None:
    try:
        current_index = segments.index(current)
    except ValueError:
        return None
    for segment in segments[current_index + 1 :]:
        if segment.start > current.start:
            return segment.start
    return None


def _split_into_subtitle_beats(segments: list[Segment], words_per_subtitle: int = 1) -> list[Segment]:
    beats: list[Segment] = []
    group_size = min(3, max(1, words_per_subtitle))
    for segment in segments:
        tokens = _tokenize_spoken_units(segment.text)
        if not tokens:
            continue
        if len(tokens) == 1 and segment.end > segment.start:
            beats.append(segment)
            continue

        if group_size == 1:
            grouped_tokens = [[token] for token in tokens]
        else:
            grouped_tokens = [tokens[index : index + group_size] for index in range(0, len(tokens), group_size)]

        if len(grouped_tokens) == 1:
            beats.append(Segment(segment.start, segment.end, " ".join(grouped_tokens[0])))
            continue

        grouped_text = [" ".join(group) for group in grouped_tokens]
        total_weight = sum(_token_weight(text) for text in grouped_text)
        duration = max(0.01, segment.end - segment.start)
        cursor = segment.start
        for index, text in enumerate(grouped_text):
            if index == len(grouped_text) - 1:
                end = segment.end
            else:
                end = cursor + duration * (_token_weight(text) / total_weight)
            beats.append(Segment(cursor, end, text))
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
