from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meeting_transcriber.diarization import (DIARIZATION_VERSION, Diarizer, cluster_embeddings,
                                             label_segments, speaker_vad_parameters, speech_windows)
from meeting_transcriber.errors import CancelledError, DiarizationError
from meeting_transcriber.formatter import display_text, render_srt
from meeting_transcriber.transcriber import convert_segment
from meeting_transcriber.utils import read_json, write_json
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


def test_windows_preserve_short_silence_and_do_not_create_short_tails():
    windows = speech_windows([{"start": 0, "end": 4}, {"start": 4.3, "end": 7.8}])
    assert all(b <= 4 or a >= 4.3 for a, b in windows)
    assert all(b - a == pytest.approx(3) for a, b in windows)
    assert windows[0][0] == 0 and windows[-1][1] == 7.8


def test_speaker_vad_does_not_change_transcription_settings(config):
    original = dict(config.vad_parameters)
    options = speaker_vad_parameters(config)
    assert options["speech_pad_ms"] == 60
    assert options["min_silence_duration_ms"] == 200
    assert options["threshold"] == original["threshold"]
    assert config.vad_parameters == original
    assert speaker_vad_parameters(replace(config, vad_parameters={"speech_pad_ms": 0}))["speech_pad_ms"] == 0


def test_short_outlier_does_not_invent_an_extra_speaker():
    vectors = np.array([[0, 0, 1], [1, 0, 0], [.98, .02, 0], [0, 1, 0], [0, .99, .01]])
    labels = cluster_embeddings(vectors, 0, .45, np.array([.6, 3, 3, 3, 3]))
    assert labels == [-1, 0, 0, 1, 1]


def test_short_returning_voice_is_matched_to_reference():
    vectors = np.array([[.95, .05], [1, 0], [0, 1], [.05, .95]])
    assert cluster_embeddings(vectors, 0, .45, np.array([.6, 3, 3, .7])) == [0, 0, 1, 1]


def test_ambiguous_short_voice_is_unknown_not_a_third_person():
    vectors = np.array([[1, 0], [0, 1], [1, 1]])
    assert cluster_embeddings(vectors, 0, .45, np.array([3, 3, .6])) == [0, 1, -1]


def test_single_reference_and_only_short_recordings_are_supported():
    vectors = np.array([[1, 0], [.99, .01], [0, 1]])
    assert cluster_embeddings(vectors, 0, .45, np.array([3, .7, .6])) == [0, 0, -1]
    assert cluster_embeddings(vectors, 0, .45, np.array([.6, .7, .6])) == [0, 0, 1]
    assert cluster_embeddings(np.empty((0, 2)), 0, .45, np.array([])) == []


def test_pooled_voice_merges_scattered_clusters_but_preserves_other_voice():
    # Two vocal conditions with per-window variation: pairwise clustering alone
    # splits them, while the pooled voice directions are close. A distinct voice
    # remains separate, and an explicit count must bypass automatic merging.
    vectors = np.zeros((9, 11))
    vectors[:4, 0] = np.sqrt(.6)
    vectors[4:8, :2] = np.sqrt(.6) * np.array([.8, .6])
    for i in range(8):
        vectors[i, i + 2] = np.sqrt(.4)
    vectors[8, 10] = 1
    assert cluster_embeddings(vectors, 0, .45) == [0] * 8 + [1]
    assert len(set(cluster_embeddings(vectors, 3, .45))) == 3


def test_explicit_count_includes_short_speakers():
    vectors = np.eye(3)
    assert cluster_embeddings(vectors, 3, .45, np.array([3, .6, .7])) == [0, 1, 2]


def test_voice_merge_does_not_chain_dissimilar_endpoints():
    vectors = np.zeros((12, 14))
    for group, angle in enumerate(np.deg2rad([0, 40, 80])):
        for index in range(group * 4, (group + 1) * 4):
            vectors[index, :2] = np.sqrt(.6) * np.array([np.cos(angle), np.sin(angle)])
            vectors[index, index + 2] = np.sqrt(.4)
    labels = cluster_embeddings(vectors, 0, .45)
    assert len(set(labels)) == 2
    assert labels[0] != labels[-1]


def test_long_recording_assignment_keeps_returning_speakers():
    vectors = np.tile(np.eye(2), (1100, 1))
    labels = cluster_embeddings(vectors, 0, .45, np.full(2200, 3))
    assert labels == [0, 1] * 1100


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


def test_alignment_uses_overlap_instead_of_nearest_center(config):
    segments = [{"id": 0, "start": 4.4, "end": 4.6, "text": "文字"}]
    windows = [{"start": 4, "end": 4.2, "speaker": 1}, {"start": 0, "end": 10, "speaker": 0}]
    assert label_segments(segments, windows, config)[0]["speaker"] == "SPEAKER_01"


@pytest.mark.parametrize("windows", [
    [{"start": 0, "end": 1, "speaker": 0}],
    [{"start": 1, "end": 2, "speaker": -1}],
    [{"start": 1, "end": 2, "speaker": 0}, {"start": 1, "end": 2, "speaker": 1}],
])
def test_uncovered_unknown_or_tied_words_are_not_guessed(config, windows):
    segments = [{"id": 0, "start": 1.1, "end": 1.5, "text": "文字"}]
    assert label_segments(segments, windows, config)[0]["speaker"] == "UNKNOWN"


def test_short_interjection_is_preserved(config):
    words = [{"start": 0, "end": 1, "word": "請問"}, {"start": 1.2, "end": 1.6, "word": "對"},
             {"start": 1.8, "end": 3, "word": "謝謝"}]
    segments = [{"id": 0, "start": 0, "end": 3, "text": "請問對謝謝", "words": words}]
    windows = [{"start": w["start"], "end": w["end"], "speaker": s} for w, s in zip(words, [0, 1, 0])]
    result = label_segments(segments, windows, config)
    assert [s["speaker"] for s in result] == ["SPEAKER_01", "SPEAKER_02", "SPEAKER_01"]
    assert "".join(s["raw_text"] for s in result) == "請問對謝謝"


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


def test_upgrade_reanalyzes_old_diarization_without_rerunning_asr(monkeypatch, tmp_path, config):
    source, worker, extractions = prepare(monkeypatch, tmp_path, replace(config, diarize=True), 30)
    calls = []
    def diarize(self, source, folder, segments):
        calls.append(1)
        return [dict(s, speaker="SPEAKER_01") for s in segments]
    monkeypatch.setattr(Diarizer, "run", diarize)
    result = worker.transcribe(source)
    metadata = read_json(result.folder / "metadata.json")
    metadata["export_options"].pop("diarization_version")
    write_json(result.folder / "metadata.json", metadata)
    assert not worker.transcribe(source).skipped
    assert len(calls) == 2 and len(extractions) == 1
    assert read_json(result.folder / "metadata.json")["export_options"]["diarization_version"] == DIARIZATION_VERSION
    assert worker.transcribe(source).skipped


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
