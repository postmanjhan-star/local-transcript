from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meeting_transcriber.diarization import Diarizer, cluster_embeddings, label_segments, speech_windows
from meeting_transcriber.errors import CancelledError, DiarizationError
from meeting_transcriber.formatter import display_text, render_srt
from meeting_transcriber.transcriber import convert_segment
from meeting_transcriber.utils import read_json
from conftest import fake_segment
from test_transcriber import prepare


def test_global_clustering_keeps_returning_speakers():
    vectors = np.array([[1, 0, 0], [.98, .02, 0], [0, 1, 0], [.01, .99, 0], [.99, .01, 0]])
    assert cluster_embeddings(vectors, 2, .45) == [0, 0, 1, 1, 0]
    assert cluster_embeddings(vectors, 0, .45) == [0, 0, 1, 1, 0]
    assert cluster_embeddings(vectors, 1, .45) == [0] * 5


def test_windows_skip_long_silence_and_bound_memory():
    windows = speech_windows([{"start": 0, "end": 7200, "words": [
        {"start": 1, "end": 5}, {"start": 7190, "end": 7199}]}])
    assert windows and all(b - a <= 3.001 for a, b in windows)
    assert not any(10 < a < 7180 for a, _ in windows)
    assert windows[-1][1] == 7199


def test_word_alignment_splits_speakers_without_losing_raw_text(config):
    words = [{"start": 0, "end": 2, "word": "这是软件。"}, {"start": 3, "end": 5, "word": "请确认。"}]
    segments = [{"id": 0, "start": 0, "end": 5, "text": "這是軟體。請確認。", "raw_text": "这是软件。请确认。", "words": words}]
    windows = [{"start": 0, "end": 2, "speaker": 0}, {"start": 3, "end": 5, "speaker": 1}]
    result = label_segments(segments, windows, config)
    assert [s["speaker"] for s in result] == ["SPEAKER_01", "SPEAKER_02"]
    assert "".join(s["raw_text"] for s in result) == segments[0]["raw_text"]
    assert result[0]["text"] == "這是軟體。"
    assert "【說話者 1】" in render_srt(result)
    assert display_text(result[1]) == "【說話者 2】請確認。"


def test_old_checkpoint_retains_text_and_unknown_is_explicit(config):
    segments = [{"id": 0, "start": 0, "end": 5, "text": "完整逐字稿"}]
    result = label_segments(segments, [], config)
    assert result[0]["text"] == "完整逐字稿" and result[0]["speaker"] == "UNKNOWN"


def test_speaker_settings_do_not_invalidate_asr_cache(config):
    assert config.fingerprint() == replace(config, diarize=True, num_speakers=2).fingerprint()


def test_toggle_diarization_reuses_asr(monkeypatch, tmp_path, config):
    source, worker, extractions = prepare(monkeypatch, tmp_path, config, 30)
    result = worker.transcribe(source)
    monkeypatch.setattr(Diarizer, "run", lambda self, source, folder, segments: [dict(s, speaker="SPEAKER_01") for s in segments])
    worker.config = replace(config, diarize=True)
    assert not worker.transcribe(source).skipped
    assert len(extractions) == 1
    assert "【說話者 1】" in (result.folder / f"{source.stem}.txt").read_text(encoding="utf-8")
    assert worker.transcribe(source).skipped
    worker.config = config
    worker.transcribe(source)
    assert len(extractions) == 1
    assert "說話者" not in (result.folder / f"{source.stem}.txt").read_text(encoding="utf-8")


def test_diarization_failure_keeps_plain_outputs(monkeypatch, tmp_path, config):
    source, worker, _ = prepare(monkeypatch, tmp_path, replace(config, diarize=True), 30)
    def failure(*args):
        raise DiarizationError("model unavailable")
    monkeypatch.setattr(Diarizer, "run", failure)
    with pytest.raises(DiarizationError):
        worker.transcribe(source)
    folder = config.output_dir / source.stem
    assert (folder / f"{source.stem}.txt").read_text(encoding="utf-8")
    assert (folder / "transcript.partial.jsonl").is_file()
    assert not read_json(folder / "metadata.json")["completed"]


def test_cancellation_is_cooperative_and_resumable(monkeypatch, tmp_path, config):
    source, worker, _ = prepare(monkeypatch, tmp_path, config, 600)
    cancelled = [False]
    worker.cancelled = lambda: cancelled[0]
    worker.on_segment = lambda s: cancelled.__setitem__(0, True)
    with pytest.raises(CancelledError):
        worker.transcribe(source)
    folder = config.output_dir / source.stem
    assert read_json(folder / "metadata.json")["status"] == "interrupted"
    assert (folder / "transcript.partial.jsonl").read_text(encoding="utf-8")
    worker.cancelled = lambda: False
    worker.on_segment = lambda s: None
    assert worker.transcribe(source).duration == 600


def test_offline_speaker_model_never_downloads(config, monkeypatch):
    engine = Diarizer(replace(config, offline=True), lambda s: None, lambda e: None, lambda: False)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("must not use network"))
    with pytest.raises(DiarizationError, match="離線"):
        engine.model_path()


def test_isolated_worker_and_cancellation(tmp_path, config):
    import wave
    source = tmp_path / "中文 靜音.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 32000)
    segments = [{"id": 0, "start": 0, "end": 2, "text": "保留文字"}]
    events = []
    engine = Diarizer(replace(config, offline=True), lambda s: None, events.append, lambda: False)
    result = engine.run(source, tmp_path, segments)
    assert result[0]["text"] == "保留文字" and result[0]["speaker"] == "UNKNOWN"
    assert any(e.phase == "diarizing" for e in events)
    assert list((tmp_path / "speaker-checkpoints").rglob("*.json"))
    stopped = [False]
    engine.cancelled = lambda: stopped[0]
    engine.progress = lambda event: stopped.__setitem__(0, True)
    with pytest.raises(CancelledError):
        engine.run(source, tmp_path, segments)
