import subprocess
import wave

import pytest

from meeting_transcriber.audio import discover_inputs, extract_chunk, find_ffmpeg, probe_audio
from meeting_transcriber.errors import AudioError, FFmpegMissingError


def test_real_m4a_unicode_and_seek(tmp_path):
    source = tmp_path / "中文 錄音.wav"
    m4a = tmp_path / "中文 錄音.m4a"
    clip = tmp_path / "分段.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x00" * 16000 * 65)
    ffmpeg = find_ffmpeg()
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(source), "-c:a", "aac", str(m4a)], check=True)
    info = probe_audio(m4a)
    assert abs(info.duration - 65) < 0.2
    assert abs(extract_chunk(ffmpeg, m4a, clip, 61, 3, info.stream_index) - 3) < 0.1


def test_discovery_and_bad_audio(tmp_path):
    (tmp_path / "a.M4A").write_bytes(b"invalid")
    (tmp_path / "ignore.txt").write_bytes(b"invalid")
    nested = tmp_path / "child"
    nested.mkdir()
    (nested / "b.mp3").write_bytes(b"invalid")
    assert len(discover_inputs(tmp_path)) == 1
    assert len(discover_inputs(tmp_path, True)) == 2
    assert len(discover_inputs(tmp_path, True, (nested,))) == 1
    assert len(discover_inputs(tmp_path / "ignore.txt")) == 1
    with pytest.raises(AudioError):
        probe_audio(tmp_path / "a.M4A")
    with pytest.raises(AudioError, match="找不到"):
        discover_inputs(tmp_path / "missing")
    with pytest.raises(FFmpegMissingError):
        find_ffmpeg(str(tmp_path / "missing.exe"))

