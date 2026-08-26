from app.srt_pipeline import (
    Segment,
    _correct_timing,
    _friendly_gemini_error,
    _load_json,
    _offset_segments,
    _segments_from_payload,
    _split_into_subtitle_beats,
    _to_srt,
)


def test_split_space_separated_khmer_units():
    beats = _split_into_subtitle_beats([Segment(0, 2.5, "ជួយ មាន អត់ មាន ឃើញ")])

    assert [beat.text for beat in beats] == ["ជួយ", "មាន", "អត់", "មាន", "ឃើញ"]
    assert beats[0].start == 0
    assert beats[-1].end == 2.5


def test_preserves_english_tokens():
    beats = _split_into_subtitle_beats([Segment(0, 1.2, "subscribe channel នេះ")])

    assert [beat.text for beat in beats] == ["subscribe", "channel", "នេះ"]


def test_corrected_srt_is_short_beat_style():
    beats = _split_into_subtitle_beats([Segment(0, 2.0, "hello បង")])
    corrected = _correct_timing(beats, duration=2.0)
    srt = _to_srt(corrected)

    assert "hello\n\n2" in srt
    assert "បង" in srt


def test_fast_speech_keeps_timing_inside_original_window():
    beats = _split_into_subtitle_beats([Segment(0, 0.7, "ជួយ មាន អត់ មាន ឃើញ")])
    corrected = _correct_timing(beats, duration=0.7)

    assert len(corrected) == 5
    assert corrected[-1].end <= 0.7
    assert all((segment.end - segment.start) <= 0.35 for segment in corrected)


def test_can_group_two_or_three_words_per_subtitle():
    two_word_beats = _split_into_subtitle_beats([Segment(0, 2, "ជួយ មាន អត់ មាន ឃើញ")], 2)
    three_word_beats = _split_into_subtitle_beats([Segment(0, 2, "one two three four")], 3)

    assert [beat.text for beat in two_word_beats] == ["ជួយ មាន", "អត់ មាន", "ឃើញ"]
    assert [beat.text for beat in three_word_beats] == ["one two three", "four"]


def test_load_json_extracts_object_from_extra_text():
    payload = _load_json('noise before {"segments":[{"start":0,"end":1,"text":"ok"}]} noise after')

    assert payload["segments"][0]["text"] == "ok"


def test_prefers_word_timings_from_gemini_payload():
    payload = {
        "segments": [
            {
                "start": 0,
                "end": 2,
                "text": "ខ្ញុំស្រលាញ់បង",
                "words": [
                    {"start": 0, "end": 0.4, "text": "ខ្ញុំ"},
                    {"start": 0.4, "end": 1.4, "text": "ស្រលាញ់"},
                    {"start": 1.4, "end": 2, "text": "បង"},
                ],
            }
        ]
    }

    segments = _segments_from_payload(payload)
    beats = _split_into_subtitle_beats(segments, 1)

    assert [beat.text for beat in beats] == ["ខ្ញុំ", "ស្រលាញ់", "បង"]
    assert beats[-1].end == 2


def test_offsets_segments_from_later_audio_chunks():
    segments = _offset_segments([Segment(0.2, 1.4, "YouTube"), Segment(1.4, 2.0, "បង")], 110)

    assert segments[0] == Segment(110.2, 111.4, "YouTube")
    assert segments[1] == Segment(111.4, 112.0, "បង")


def test_friendly_error_explains_api_key_problem():
    message = _friendly_gemini_error(Exception("API key not valid"), ["gemini-2.5-flash"])

    assert "API key" in message
    assert "/setgemini" in message


def test_friendly_error_explains_unsupported_model():
    message = _friendly_gemini_error(Exception("404 model not found"), ["gemini-3.6-flash", "gemini-2.5-flash"])

    assert "Model AI" in message
    assert "gemini-2.5-flash" in message
