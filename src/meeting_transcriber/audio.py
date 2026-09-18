"""Probe containers with PyAV and extract bounded audio windows with FFmpeg."""
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
import wave

from .errors import AudioError, FFmpegMissingError

SUPPORTED_EXTENSIONS = frozenset({".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".mov", ".webm",
                                  ".aac", ".wma", ".mkv", ".opus", ".aiff", ".aif", ".mpeg", ".mpg"})


@dataclass(frozen=True)
class AudioInfo:
    duration: float
    stream_index: int


def find_ffmpeg(explicit: str | None = None) -> str:
    executable = explicit or shutil.which("ffmpeg")
    if not executable:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise FFmpegMissingError("FFmpeg 尚未安裝。請執行 setup.ps1 或安裝 imageio-ffmpeg。") from exc
    try:
        subprocess.run([str(executable), "-version"], check=True, capture_output=True,
                       timeout=15, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.SubprocessError) as exc:
        raise FFmpegMissingError(f"FFmpeg 尚未安裝或無法啟動：{executable}") from exc
    return str(executable)


def probe_audio(source: Path) -> AudioInfo:
    if not source.is_file():
        raise AudioError(f"找不到輸入檔案：{source}")
    try:
        import av
        with av.open(str(source)) as container:
            if not container.streams.audio:
                raise ValueError("沒有音軌")
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            else:
                # Unindexed containers: scan packets without decoding or buffering audio.
                duration = 0.0
                origin = float((stream.start_time or 0) * stream.time_base) if stream.time_base else 0.0
                for packet in container.demux(stream):
                    if packet.pts is not None and packet.time_base:
                        duration = max(duration, float((packet.pts + (packet.duration or 0)) * packet.time_base) - origin)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("無法取得有效音訊長度")
            return AudioInfo(duration, stream.index)
    except (ImportError, OSError, ValueError) as exc:
        raise AudioError(f"無法解析音訊格式：{source.name}（{exc}）") from exc


def extract_chunk(executable: str, source: Path, destination: Path, start: float,
                  duration: float, stream_index: int) -> float:
    command = [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
               "-map", f"0:{stream_index}", "-vn", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", str(destination)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=max(180, duration * 4),
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        with wave.open(str(destination), "rb") as audio:
            actual = audio.getnframes() / audio.getframerate()
        if actual <= 0 or actual < duration - 1.0:
            raise AudioError(f"音軌提早結束（預期 {duration:.1f} 秒，實際 {actual:.1f} 秒），已保留 checkpoint。")
        return actual
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()[-1500:]
        raise AudioError(f"無法解析音訊格式：{source.name}（FFmpeg：{detail}）") from exc
    except (OSError, subprocess.TimeoutExpired, wave.Error) as exc:
        raise AudioError(f"音訊分段失敗：{source.name}（{exc}）") from exc


def discover_inputs(path: Path, recursive: bool = False, excluded: tuple[Path, ...] = ()) -> list[Path]:
    if path.is_file():
        return [path.resolve()]  # Explicit files may use any FFmpeg-supported extension.
    if not path.is_dir():
        raise AudioError(f"找不到輸入檔案：{path}")
    files = path.rglob("*") if recursive else path.iterdir()
    roots = tuple(p.resolve() for p in excluded)
    return sorted((p.resolve() for p in files if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                   and not any(p.resolve().is_relative_to(root) for root in roots)),
                  key=lambda p: str(p).casefold())

