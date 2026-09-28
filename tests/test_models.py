from dataclasses import replace
from types import SimpleNamespace

import pytest

from meeting_transcriber.errors import ModelDownloadError
from meeting_transcriber.models import ModelManager


@pytest.mark.parametrize("cuda_count,device,compute", [(1, "cuda", "float16"), (0, "cpu", "int8")])
def test_auto_device_selection_with_diarization_enabled(monkeypatch, config, cuda_count, device, compute):
    import ctranslate2
    import faster_whisper
    calls = []
    manager = ModelManager(replace(config, diarize=True), lambda message: None)
    monkeypatch.setattr(manager, "resolve_path", lambda: "local-model")
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: cuda_count)
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda device: {"float16"})
    def load(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(faster_whisper, "WhisperModel", load)
    manager.ensure_loaded()
    assert len(calls) == 1
    assert (calls[0]["device"], calls[0]["compute_type"]) == (device, compute)
    assert (manager.device, manager.compute_type) == (device, compute)


def test_cuda_init_failure_falls_back(monkeypatch, config):
    import ctranslate2
    import faster_whisper
    calls, warnings = [], []
    manager = ModelManager(config, warnings.append)
    monkeypatch.setattr(manager, "resolve_path", lambda: "local-model")
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda device: {"float16"})
    def load(path, **kwargs):
        calls.append(kwargs)
        if kwargs["device"] == "cuda":
            raise RuntimeError("CUDA initialization failed")
        return SimpleNamespace()
    monkeypatch.setattr(faster_whisper, "WhisperModel", load)
    manager.ensure_loaded()
    assert [call["device"] for call in calls] == ["cuda", "cpu"]
    assert calls[-1]["compute_type"] == "int8" and manager.device == "cpu"
    assert warnings


def test_cache_first_never_downloads_complete_model(monkeypatch, tmp_path, config):
    import faster_whisper.utils
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / name).write_text("test")
    flags = []
    def download(model, **kwargs):
        flags.append(kwargs["local_files_only"])
        return str(tmp_path)
    monkeypatch.setattr(faster_whisper.utils, "download_model", download)
    manager = ModelManager(config, lambda message: None)
    assert manager.resolve_path() == str(tmp_path)
    assert flags == [True]


def test_offline_missing_does_not_download(monkeypatch, config):
    import faster_whisper.utils
    flags = []
    def download(model, **kwargs):
        flags.append(kwargs["local_files_only"])
        raise FileNotFoundError("not cached")
    monkeypatch.setattr(faster_whisper.utils, "download_model", download)
    manager = ModelManager(replace(config, offline=True), lambda message: None)
    with pytest.raises(ModelDownloadError):
        manager.resolve_path()
    assert flags == [True]

