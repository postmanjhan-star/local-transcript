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


def test_asr_hints_validated_and_fingerprinted(tmp_path):
    from dataclasses import replace
    path = tmp_path / "config.yaml"
    path.write_text("hotwords: 'SMT MES 料站表'\ncondition_on_previous_text: true\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.hotwords == "SMT MES 料站表"
    assert cfg.condition_on_previous_text
    assert cfg.fingerprint() != replace(cfg, hotwords="").fingerprint()
    assert cfg.fingerprint() != replace(cfg, condition_on_previous_text=False).fingerprint()
    for invalid in ("hotwords: []", "condition_on_previous_text: 'true'"):
        path.write_text(invalid, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


def test_automatic_glossary_and_private_terms(tmp_path):
    from meeting_transcriber.config import effective_hotwords, hotword_terms
    path = tmp_path / "config.yaml"
    path.write_text(
        "hotwords: 'Panasonic, Reel ID'\n"
        "vocabulary: [SMT, Panasonic]\n"
        "terminology: {iQube: iCube, waffer: Wafer}\n",
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.local.txt").write_text(
        "# This stays on the device\nWafer\nSMT\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert hotword_terms(cfg) == ["Panasonic", "Reel ID", "SMT", "Wafer", "iCube"]
    assert effective_hotwords(cfg) == "Panasonic, Reel ID, SMT, Wafer, iCube"
    assert cfg.fingerprint() != replace(cfg, vocabulary=[]).fingerprint()


def test_keyword_parser_does_not_split_letter_n(tmp_path):
    from meeting_transcriber.config import hotword_terms
    path = tmp_path / "config.yaml"
    path.write_text("hotwords: Panasonic\n", encoding="utf-8")
    cfg = load_config(path)
    assert hotword_terms(cfg) == ["Panasonic"]


@pytest.mark.parametrize("invalid", ["vocabulary: anything", "vocabulary: [ok, '']", "vocabulary: [123]"])
def test_reject_invalid_vocabulary(tmp_path, invalid):
    path = tmp_path / "config.yaml"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
