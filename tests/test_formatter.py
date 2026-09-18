from types import SimpleNamespace

import pytest

from meeting_transcriber.formatter import (TextNormalizer, render_markdown, render_srt,
                                           suspicious_reasons, timestamp, write_outputs)
from meeting_transcriber.transcriber import convert_segment
from conftest import fake_segment


@pytest.mark.parametrize("seconds,plain,srt", [
    (0, "00:00:00", "00:00:00,000"), (9.1, "00:00:09", "00:00:09,100"),
    (3599.9999, "00:59:59", "01:00:00,000"), (7201.123, "02:00:01", "02:00:01,123"),
])
def test_timestamps(seconds, plain, srt):
    assert timestamp(seconds) == plain
    assert timestamp(seconds, True) == srt


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_timestamp(value):
    with pytest.raises(ValueError):
        timestamp(value)


def test_srt():
    assert render_srt([{"start": 3, "end": 9.1, "text": "目前 Wafer 有記號。"},
                       {"start": 10, "end": 15.2, "text": "第二句。"}]) == (
        "1\n00:00:03,000 --> 00:00:09,100\n目前 Wafer 有記號。\n\n"
        "2\n00:00:10,000 --> 00:00:15,200\n第二句。\n")


def test_terminology_boundaries_and_non_cascading():
    normalizer = TextNormalizer(False, {"iq cube": "iCube", "wafer": "Wafer", "pcb": "PCB", "ERP": "X", "X": "Y"})
    assert normalizer.normalize("IQ CUBE wafer pcb ERP X") == "iCube Wafer PCB X Y"
    assert normalizer.normalize("wafer123 superpcb erp_id 這是wafer。") == "wafer123 superpcb erp_id 這是Wafer。"
    assert normalizer.normalize("一般中文不猜測") == "一般中文不猜測"


def test_traditional_conversion():
    normalizer = TextNormalizer(True, {"iQube": "iCube"})
    assert normalizer.normalize("软件 材料 批号 iQube API PCB-A01") == "軟體 材料 批號 iCube API PCB-A01"
    assert TextNormalizer(False, {}).normalize("软件") == "软件"


def test_markdown_groups_and_prompt():
    metadata = {"file": "新錄音.m4a", "duration": 7200, "model": "medium", "language": "zh"}
    text = render_markdown("新錄音", metadata, [{"start": 2, "text": "一"}, {"start": 600, "text": "二"},
                                               {"start": 3600, "text": "三"}])
    assert "## 00:00–10:00" in text and "## 10:00–20:00" in text and "## 60:00–70:00" in text
    assert "不得自行新增" in text and "批次／批號" in text


def test_overlap_uses_words_and_keeps_raw():
    words = [SimpleNamespace(start=0, end=2, word="旧句。"), SimpleNamespace(start=2, end=5, word="新句软件。")]
    segment = fake_segment(0, 5, "旧句。新句软件。", words)
    converted = convert_segment(segment, 10, 12, 30, TextNormalizer(True, {}), [])
    assert converted["start"] == 12 and converted["end"] == 15
    assert converted["raw_text"] == "新句软件。"
    assert converted["text"] == "新句軟體。"
    assert converted["original_segment_text"] == "旧句。新句软件。"
    assert convert_segment(segment, 10, 15, 30, TextNormalizer(False, {}), []) is None


def test_suspicious_preserves_single_normal_phrase():
    assert suspicious_reasons("謝謝觀看", []) == []
    previous = [{"text": "謝謝觀看"}] * 2
    assert "possible_outro_hallucination" in suspicious_reasons("謝謝觀看", previous)


def test_utf8_outputs(tmp_path, config):
    metadata = {"file": "新錄音.m4a", "duration": 12, "model": "medium", "language": "zh"}
    segments = [{"id": 0, "start": 3, "end": 9.1, "text": "批次與晶圓。", "raw_text": "批次与晶圆。"}]
    outputs = write_outputs(tmp_path, "新錄音", metadata, segments, config)
    assert len(outputs) == 5
    for name in outputs:
        assert "批次與晶圓。" in (tmp_path / name).read_text(encoding="utf-8")
    assert "00:00:03" not in (tmp_path / "新錄音.txt").read_text(encoding="utf-8")

