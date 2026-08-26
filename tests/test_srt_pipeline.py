from app.srt_pipeline import Segment, _correct_timing, _split_into_subtitle_beats, _to_srt


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
