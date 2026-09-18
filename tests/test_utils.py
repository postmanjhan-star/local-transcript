from pathlib import Path

import pytest

from meeting_transcriber.errors import CheckpointError, TranscriberError
from meeting_transcriber.utils import (already_processed, append_segment, archive_results, atomic_write,
                                       load_partial, output_lock, output_path, source_identity, write_json)


def test_already_processed(tmp_path):
    source = tmp_path / "會議.m4a"
    source.write_bytes(b"audio")
    identity = source_identity(source)
    metadata = {**identity, "completed": True, "config_hash": "abc"}
    assert already_processed(metadata, identity, "abc")
    assert not already_processed(metadata, identity, "xyz")
    assert not already_processed({**metadata, "completed": False}, identity, "abc")
    source.write_bytes(b"changed audio")
    assert not already_processed(metadata, source_identity(source), "abc")


def test_output_collision_and_chinese(tmp_path):
    source = tmp_path / "新錄音.m4a"
    source.write_bytes(b"audio")
    root = tmp_path / "output"
    first = output_path(source, root)
    assert first == root / "新錄音"
    write_json(first / "metadata.json", source_identity(source))
    assert output_path(source, root) == first
    assert output_path(tmp_path / "新錄音.mp3", root) != first
    assert output_path(tmp_path / "other" / "新錄音.m4a", root) != first


def test_torn_checkpoint_and_nonterminated_json(tmp_path):
    path = tmp_path / "transcript.partial.jsonl"
    segment = {"start": 1, "end": 2, "text": "已保存"}
    append_segment(path, segment)
    with path.open("ab") as stream:
        stream.write(b'{"start":3,"text":"\xe4')
    assert load_partial(path) == [segment]
    assert len(list(tmp_path.glob("*.damaged-*"))) == 1
    append_segment(path, {"start": 3, "end": 4, "text": "下句"})
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert len(load_partial(path)) == 2 and path.read_bytes().endswith(b"\n")


def test_corrupt_checkpoint_not_silently_lost(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text('broken\n{"start":1,"end":2,"text":"good"}\n', encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(CheckpointError):
        load_partial(path)
    assert path.read_bytes() == original


def test_force_archive_retains_results(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    atomic_write(root / "舊逐字稿.txt", "保留")
    archive = archive_results(root)
    assert archive.is_relative_to(root)
    assert (archive / "舊逐字稿.txt").read_text(encoding="utf-8") == "保留"
    assert not (root / "舊逐字稿.txt").exists()


def test_output_lock_releases(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(TranscriberError):
            with output_lock(tmp_path):
                pass
    with output_lock(tmp_path):
        pass


def test_atomic_failure_preserves_previous(tmp_path, monkeypatch):
    path = tmp_path / "output.txt"
    atomic_write(path, "old")
    def fail_replace(*args):
        raise OSError("disk failure")
    monkeypatch.setattr("meeting_transcriber.utils.os.replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write(path, "new")
    assert path.read_text(encoding="utf-8") == "old"

