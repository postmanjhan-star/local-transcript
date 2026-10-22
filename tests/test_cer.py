"""CER evaluation does not need an ASR model or network."""
import importlib.util
from pathlib import Path
import pytest

file = Path(__file__).resolve().parents[1] / "scripts" / "measure_cer.py"
spec = importlib.util.spec_from_file_location("measure_cer", file)
cer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cer)


def test_normalization():
    assert cer.measure("我們討論 SMT。", "我們討論 smt！")["cer"] == 0


def test_substitution_and_deletion():
    assert cer.measure("你好世界", "你好世")["cer"] == 0.25
    assert cer.measure("中文", "空文")["cer"] == 0.5


def test_json_raw_and_normalized(tmp_path):
    output = tmp_path / "meeting.json"
    output.write_text('{"segments":[{"raw_text":"iQube","text":"iCube"}]}', encoding="utf-8")
    assert cer.read_transcript(output, raw=True) == "iQube"
    assert cer.read_transcript(output) == "iCube"


def test_empty_reference_rejected():
    with pytest.raises(ValueError, match="空白"):
        cer.measure("，！", "有內容")
