from dataclasses import replace
from types import SimpleNamespace

import pytest

from conftest import fake_segment
from meeting_transcriber.audio import AudioInfo
from meeting_transcriber.transcriber import Transcriber
from meeting_transcriber.utils import load_partial, read_json


def prepare(monkeypatch, tmp_path, config, duration=7200):
    source = tmp_path / "新錄音.m4a"
    source.write_bytes(b"test-audio")
    monkeypatch.setattr("meeting_transcriber.transcriber.probe_audio", lambda p: AudioInfo(duration, 0))
    monkeypatch.setattr("meeting_transcriber.transcriber.find_ffmpeg", lambda p: "ffmpeg")
    extractions = []
    def extract(ffmpeg, src, dest, offset, length, index):
        extractions.append((offset, length))
        return length
    monkeypatch.setattr("meeting_transcriber.transcriber.extract_chunk", extract)
    worker = Transcriber(config, lambda message: None)
    def model_transcribe(*args, **kwargs):
        offset, length = extractions[-1]
        # Both start and end regions contain speech; the old one should be deduplicated.
        return iter([fake_segment(1, 3), fake_segment(length - 4, length - 1)]), None
    model = SimpleNamespace(transcribe=model_transcribe)
    worker.manager.model = model
    worker.manager.device = "cpu"
    worker.manager.compute_type = "int8"
    return source, worker, extractions


def test_full_120_minutes_and_skip(monkeypatch, tmp_path, config):
    source, worker, extractions = prepare(monkeypatch, tmp_path, config)
    result = worker.transcribe(source)
    assert result.duration == 7200 and len(extractions) == 24
    assert max(length for _, length in extractions) <= 315
    assert result.segment_count >= 24
    partial = load_partial(result.folder / "transcript.partial.jsonl")
    assert partial[-1]["end"] == 7199
    assert all(s["start"] >= p["end"] for p, s in zip(partial, partial[1:]))
    assert len(result.generated) == 5
    assert read_json(result.folder / "metadata.json")["completed"] is True
    assert worker.transcribe(source).skipped
    assert len(extractions) == 24


def test_crash_resumes_without_losing_saved_text(monkeypatch, tmp_path, config):
    source, worker, extractions = prepare(monkeypatch, tmp_path, config, 900)
    original = worker.manager.model.transcribe
    def interrupted(*args, **kwargs):
        if len(extractions) == 2:
            def iterator():
                yield fake_segment(20, 30, "这段已保存。")
                raise RuntimeError("Simulated interruption")
            return iterator(), None
        return original(*args, **kwargs)
    worker.manager.model.transcribe = interrupted
    with pytest.raises(RuntimeError, match="Simulated"):
        worker.transcribe(source)
    folder = config.output_dir / source.stem
    saved = load_partial(folder / "transcript.partial.jsonl")
    assert saved[-1]["end"] == 315
    assert not read_json(folder / "metadata.json")["completed"]
    worker.manager.model.transcribe = original
    count = len(extractions)
    result = worker.transcribe(source)
    assert extractions[count][0] == 300  # 315-second watermark minus 15 seconds.
    assert load_partial(folder / "transcript.partial.jsonl")[:len(saved)] == saved
    assert read_json(result.folder / "metadata.json")["completed"]


def test_silence_checkpoint_survives_crash(monkeypatch, tmp_path, config):
    source, worker, extractions = prepare(monkeypatch, tmp_path, config, 900)
    def silence(*args, **kwargs):
        if len(extractions) == 2:
            raise RuntimeError("crash after silent window")
        return iter([]), None
    worker.manager.model.transcribe = silence
    with pytest.raises(RuntimeError):
        worker.transcribe(source)
    worker.manager.model.transcribe = lambda *a, **k: (iter([]), None)
    worker.transcribe(source)
    assert extractions[2][0] == 285


def test_runtime_cuda_failure_keeps_committed_segments(monkeypatch, tmp_path, config):
    source, worker, extractions = prepare(monkeypatch, tmp_path, config, 30)
    worker.manager.device = "cuda"
    worker.manager.compute_type = "float16"
    def broken(*args, **kwargs):
        def iterator():
            yield fake_segment(1, 3)
            raise RuntimeError("CUDA failed")
        return iterator(), None
    worker.manager.model.transcribe = broken
    def fallback():
        worker.manager.device, worker.manager.compute_type = "cpu", "int8"
        worker.manager.model = SimpleNamespace(transcribe=lambda *a, **k: (iter([fake_segment(1, 3), fake_segment(10, 12)]), None))
    monkeypatch.setattr(worker.manager, "fallback_cpu", fallback)
    result = worker.transcribe(source)
    assert result.segment_count == 2
    data = read_json(result.folder / f"{source.stem}.json")
    assert data["device"] == "cpu" and len(data["device_history"]) == 2


def test_missing_output_regenerates_without_model(monkeypatch, tmp_path, config):
    source, worker, extractions = prepare(monkeypatch, tmp_path, config, 30)
    result = worker.transcribe(source)
    (result.folder / f"{source.stem}.srt").unlink()
    worker.manager.model = None
    monkeypatch.setattr(worker.manager, "ensure_loaded", lambda: pytest.fail("should not load model"))
    assert not worker.transcribe(source).skipped
    assert len(extractions) == 1
    assert (result.folder / f"{source.stem}.srt").exists()


def test_force_and_changed_source_archive(monkeypatch, tmp_path, config):
    source, worker, _ = prepare(monkeypatch, tmp_path, config, 30)
    result = worker.transcribe(source)
    first_partial = (result.folder / "transcript.partial.jsonl").read_bytes()
    worker.transcribe(source, force=True)
    backups = list((result.folder / "history").iterdir())
    assert len(backups) == 1 and (backups[0] / "transcript.partial.jsonl").read_bytes() == first_partial
    source.write_bytes(b"a changed recording")
    assert not worker.transcribe(source).skipped
    assert len(list((result.folder / "history").iterdir())) == 2


def test_export_failure_preserves_checkpoint(monkeypatch, tmp_path, config):
    source, worker, extractions = prepare(monkeypatch, tmp_path, config, 30)
    import meeting_transcriber.transcriber as module
    export = module.write_outputs
    def fail(*args, **kwargs):
        raise OSError("output full")
    monkeypatch.setattr(module, "write_outputs", fail)
    with pytest.raises(OSError):
        worker.transcribe(source)
    folder = config.output_dir / source.stem
    assert len(load_partial(folder / "transcript.partial.jsonl")) == 2
    assert not read_json(folder / "metadata.json")["completed"]
    monkeypatch.setattr(module, "write_outputs", export)
    worker.transcribe(source)
    assert len(extractions) == 1


def test_modified_source_not_marked_complete(monkeypatch, tmp_path, config):
    source, worker, _ = prepare(monkeypatch, tmp_path, config, 30)
    from meeting_transcriber.errors import TranscriberError
    def modify(*args, **kwargs):
        source.write_bytes(b"modified during recognition")
        return iter([fake_segment(1, 3)]), None
    worker.manager.model.transcribe = modify
    with pytest.raises(TranscriberError, match="修改"):
        worker.transcribe(source)
    assert not read_json(config.output_dir / source.stem / "metadata.json")["completed"]


@pytest.mark.parametrize("name", ["metadata", "checkpoint"])
def test_reserved_recording_names(monkeypatch, tmp_path, config, name):
    source, worker, _ = prepare(monkeypatch, tmp_path, config, 30)
    renamed = source.with_name(name + ".m4a")
    source.rename(renamed)
    result = worker.transcribe(renamed)
    assert read_json(result.folder / "metadata.json")["completed"]
    assert read_json(result.folder / "checkpoint.json")["processed_until"] == 30
    assert len(read_json(result.folder / f"{name}.transcript.json")["segments"]) == 2


def test_hotwords_and_context_forwarded(monkeypatch, tmp_path, config):
    config = replace(config, hotwords="iCube SMT 料站表", condition_on_previous_text=True)
    source, worker, _ = prepare(monkeypatch, tmp_path, config, 30)
    captured = {}
    def transcribe(*args, **kwargs):
        captured.update(kwargs)
        return iter([fake_segment(1, 3)]), None
    worker.manager.model.transcribe = transcribe
    worker.transcribe(source)
    assert captured["hotwords"] == "iCube SMT 料站表"
    assert captured["condition_on_previous_text"] is True
    assert captured["language"] == "zh"
