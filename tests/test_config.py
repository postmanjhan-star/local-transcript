from dataclasses import replace

import pytest

from meeting_transcriber.config import load_config
from meeting_transcriber.errors import ConfigError


def test_config_load_and_paths(tmp_path):
    path = tmp_path / "中文設定.yaml"
    path.write_text("model: small\ninput_dir: 錄音\noutput:\n  srt: false\n", encoding="utf-8-sig")
    cfg = load_config(path, {"model": "medium"})
    assert cfg.model == "medium" and cfg.language == "zh"
    assert cfg.input_dir == tmp_path / "錄音"
    assert cfg.output.txt and not cfg.output.srt


@pytest.mark.parametrize("text", ["- invalid", "42", "0", "false", "device: magic", "beam_size: 0", "beam_size: true",
    "traditional_chinese: 'false'", "chunk_seconds: .nan", "chunk_seconds: 5", "overlap_seconds: 99",
    "output: {txt: yesplease}", "output: {pdf: true}", "unknown_option: true", "terminology: []",
    "terminology: {foo: A, FOO: B}", "vad_parameters: {threshold: 1.5}",
    "vad_parameters: {mystery: 1}", "vad_parameters: {speech_pad_ms: 1.5}", "input_dir: null"])
def test_reject_invalid(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_fingerprint(config):
    assert config.fingerprint() == replace(config, offline=True, device="cpu").fingerprint()
    assert config.fingerprint() != replace(config, model="small").fingerprint()
    assert config.fingerprint() != replace(config, terminology={"wafer": "Wafer"}).fingerprint()
