"""Bounded-memory, durable transcription orchestration."""
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

from .audio import extract_chunk, find_ffmpeg, probe_audio
from .config import Config
from .errors import CancelledError, CheckpointError, TranscriberError
from .formatter import TextNormalizer, expected_outputs, suspicious_reasons, write_outputs
from .models import ModelManager
from .progress import ProgressEvent
from .utils import (already_processed, append_segment, archive_results, load_partial, now_iso,
                    output_path, read_json, same_source, source_identity, write_json)


@dataclass(frozen=True)
class TranscriptionResult:
    source: Path
    folder: Path
    duration: float
    processing_seconds: float
    segment_count: int
    generated: list[str]
    skipped: bool = False


def convert_segment(segment: Any, offset: float, watermark: float, limit: float,
                    normalizer: TextNormalizer, previous: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Drop already committed words by absolute time; never fuzzy-match content."""
    start, end = offset + float(segment.start), min(limit, offset + float(segment.end))
    if not math.isfinite(start) or not math.isfinite(end):
        raise TranscriberError("模型產生無效時間戳，已保留 checkpoint。")
    if end <= watermark + 0.001:
        return None
    original = segment.text
    raw = original
    trimmed = start < watermark - 0.001
    if trimmed and getattr(segment, "words", None):
        words = [w for w in segment.words if offset + (w.start + w.end) / 2 > watermark]
        if not words:
            return None
        raw = "".join(w.word for w in words)
        start = offset + words[0].start
        end = min(limit, offset + words[-1].end)
    start = max(0.0, watermark, start)
    if end <= start or not raw.strip():
        return None
    text = normalizer.normalize(raw)
    if not text:
        return None
    metrics = {key: float(getattr(segment, key, 0)) for key in ("avg_logprob", "no_speech_prob", "compression_ratio")}
    reasons = suspicious_reasons(text, previous, **metrics)
    if trimmed and not getattr(segment, "words", None):
        reasons.append("overlap_without_word_timestamps")
    words = [{"start": max(start, offset + float(w.start)), "end": min(end, offset + float(w.end)), "word": w.word}
             for w in (getattr(segment, "words", None) or [])
             if offset + (w.start + w.end) / 2 > watermark and offset + w.start < end]
    words = [w for w in words if w["end"] >= w["start"]]
    return {"id": len(previous), "start": round(start, 6), "end": round(end, 6), "words": words,
            "text": text, "raw_text": raw, "original_segment_text": original,
            "suspicious": bool(reasons), "suspicious_reasons": reasons, **metrics}


class Transcriber:
    def __init__(self, config: Config, warn: Callable[[str], None],
                 progress: Callable[[float, float, str], None] | None = None,
                 events: Callable[[ProgressEvent], None] | None = None,
                 cancelled: Callable[[], bool] | None = None,
                 on_segment: Callable[[dict[str, Any]], None] | None = None):
        self.config = config
        self.warn = warn
        self.progress = progress or (lambda done, total, device: None)
        self.events = events or (lambda event: None)
        self.cancelled = cancelled or (lambda: False)
        self.on_segment = on_segment or (lambda segment: None)
        self.manager = ModelManager(config, warn)
        self.normalizer = TextNormalizer(config.traditional_chinese, config.terminology)
        self.ffmpeg: str | None = None
        self.logger = logging.getLogger(__name__)

    def check_cancel(self) -> None:
        if self.cancelled():
            raise CancelledError("已停止，已完成的片段保留，下次執行會接續。")

    def report(self, done: float, total: float) -> None:
        self.progress(done, total, self.manager.device)
        self.events(ProgressEvent("transcribing", done, total, self.manager.device))

    def export_options(self) -> dict[str, Any]:
        options = {"diarize": self.config.diarize, "num_speakers": self.config.num_speakers if self.config.diarize else 0,
                   "speaker_threshold": self.config.speaker_threshold if self.config.diarize else 0}
        if self.config.diarize:
            from .diarization import DIARIZATION_VERSION
            options["diarization_version"] = DIARIZATION_VERSION
        return options

    def transcribe(self, source: Path, force: bool = False) -> TranscriptionResult:
        started = time.monotonic()
        self.check_cancel()
        self.events(ProgressEvent("preparing"))
        identity = source_identity(source)
        fingerprint = self.config.fingerprint()
        folder = output_path(source, self.config.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        metadata_path = folder / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() and not force else {}
        if not force and already_processed(metadata, identity, fingerprint):
            files = expected_outputs(folder, source.stem, self.config.output)
            previous_options = metadata.get("export_options", {"diarize": False, "num_speakers": 0, "speaker_threshold": 0})
            if all(p.is_file() for p in files.values()) and previous_options == self.export_options():
                return TranscriptionResult(source, folder, metadata["duration"], 0,
                                           metadata["segment_count"], list(p.name for p in files.values()), True)
        incompatible = bool(metadata) and (not same_source(metadata, identity) or metadata.get("config_hash") != fingerprint)
        if force or incompatible:
            backup = archive_results(folder)
            if backup:
                self.warn(f"舊結果已備份：{backup}")
            metadata = {}
        partial = folder / "transcript.partial.jsonl"
        checkpoint_path = folder / "checkpoint.json"
        if not metadata and (partial.exists() or checkpoint_path.exists()):
            raise CheckpointError(f"checkpoint 缺少來源資訊，原檔保留於 {folder}。請使用 --force 備份後重跑。")
        info = probe_audio(source)
        segments = load_partial(partial)
        checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {}
        done = float(checkpoint.get("processed_until", 0))
        if not math.isfinite(done) or done < 0 or done > info.duration + 1:
            raise CheckpointError("checkpoint 進度無效，請使用 --force 備份後重跑。")
        # A chunk checkpoint never supersedes lost segment data.
        if checkpoint.get("segment_count", 0) > len(segments):
            done = segments[-1]["end"] if segments else 0
            self.warn("checkpoint 最後一筆未完整寫入，從最後有效片段接續。")
        done = min(info.duration, max(done, segments[-1]["end"] if segments else 0))
        if done:
            self.warn(f"接續既有結果：已完成 {done:.1f} 秒，共 {len(segments)} 段。")
        metadata = {**metadata, **identity, "file": source.name, "duration": info.duration,
                    "language": self.config.language, "model": self.config.model,
                    "created_at": metadata.get("created_at", now_iso()), "config_hash": fingerprint,
                    "checkpoint_version": 1, "completed": False, "status": "running",
                    "device_history": metadata.get("device_history", [])}
        prior_processing = float(metadata.get("processing_seconds", 0))
        write_json(metadata_path, metadata)
        self.logger.info("Start source=%s model=%s duration=%.3f resume=%.3f", source,
                         self.config.model, info.duration, done)
        try:
            if done < info.duration - 0.001:
                self.events(ProgressEvent("loading", done, info.duration))
                self.ffmpeg = self.ffmpeg or find_ffmpeg(self.config.ffmpeg_path)
                self.manager.ensure_loaded()
                self.check_cancel()
                self._record_device(metadata)
                self.report(done, info.duration)
                with tempfile.TemporaryDirectory(prefix="audio-", dir=folder) as temp:
                    audio_path = Path(temp) / "chunk.wav"
                    while done < info.duration - 0.001:
                        self.check_cancel()
                        offset = max(0.0, done - self.config.overlap_seconds)
                        chunk_end = min(info.duration, done + self.config.chunk_seconds)
                        extract_chunk(self.ffmpeg, source, audio_path, offset, chunk_end - offset, info.stream_index)
                        self._transcribe_chunk(audio_path, offset, chunk_end, done, segments, partial, info.duration, metadata)
                        done = chunk_end
                        # Silence also counts as completed work, so resume needn't re-scan it.
                        write_json(checkpoint_path, {"processed_until": done, "segment_count": len(segments),
                                                     "updated_at": now_iso()})
                        self.report(done, info.duration)
            if not same_source(identity, source_identity(source)):
                raise TranscriberError("轉錄期間來源檔案被修改，未標為完成；已保留 checkpoint。")
            metadata.update({"device": self.manager.device if self.manager.device != "uninitialized" else metadata.get("device", "unknown"),
                             "compute_type": self.manager.compute_type if self.manager.device != "uninitialized" else metadata.get("compute_type", "unknown"),
                             "segment_count": len(segments), "processed_until": info.duration,
                             "processing_seconds": prior_processing + time.monotonic() - started,
                             "completed_at": now_iso()})
            exported = {k: metadata[k] for k in ("file", "duration", "language", "model", "device", "compute_type",
                                                "created_at", "device_history", "processing_seconds")}
            self.check_cancel()
            final_segments = segments
            if self.config.diarize:
                # A speaker-model failure must not hide the completed transcription.
                files = expected_outputs(folder, source.stem, self.config.output)
                if not all(p.is_file() for p in files.values()):
                    write_outputs(folder, source.stem, {**exported, "diarization_status": "pending"}, segments, self.config)
                from .diarization import Diarizer
                final_segments = Diarizer(self.config, self.warn, self.events, self.cancelled).run(source, folder, segments)
            self.check_cancel()
            if not same_source(identity, source_identity(source)):
                raise TranscriberError("分析期間來源檔案被修改，結果未標為完成。")
            self.events(ProgressEvent("writing"))
            exported.update({"diarization_enabled": self.config.diarize,
                             "speakers": sorted({s["speaker"] for s in final_segments if "speaker" in s}),
                             "processing_seconds": prior_processing + time.monotonic() - started})
            generated = write_outputs(folder, source.stem, exported, final_segments, self.config)
            metadata.update({"completed": True, "status": "completed", "generated": generated,
                             "export_options": self.export_options(), "speakers": exported["speakers"],
                             "processing_seconds": exported["processing_seconds"]})
            metadata.pop("last_error", None)
            write_json(metadata_path, metadata)
            elapsed = time.monotonic() - started
            self.logger.info("Completed source=%s device=%s compute=%s segments=%d processing=%.3f",
                             source, metadata["device"], metadata["compute_type"], len(segments), elapsed)
            return TranscriptionResult(source, folder, info.duration, elapsed, len(segments), generated)
        except BaseException as exc:
            metadata.update({"completed": False, "status": "interrupted" if isinstance(exc, (KeyboardInterrupt, CancelledError)) else "failed",
                             "last_error": str(exc), "segment_count": len(segments),
                             "processing_seconds": prior_processing + time.monotonic() - started})
            try:
                write_json(metadata_path, metadata)
            except OSError:
                self.logger.exception("Failed to save failure metadata; partial file retained")
            raise

    def _record_device(self, metadata: dict[str, Any]) -> None:
        device = {"device": self.manager.device, "compute_type": self.manager.compute_type}
        if not metadata["device_history"] or metadata["device_history"][-1] != device:
            metadata["device_history"].append(device)
        metadata.update(device)
        self.logger.info("Device=%s compute_type=%s", self.manager.device, self.manager.compute_type)

    def _transcribe_chunk(self, path: Path, offset: float, chunk_end: float, done: float,
                          segments: list[dict[str, Any]], partial: Path, duration: float,
                          metadata: dict[str, Any]) -> None:
        while True:
            self.check_cancel()
            iterator = None
            try:
                iterator, _ = self.manager.model.transcribe(
                    str(path), language=self.config.language, beam_size=self.config.beam_size,
                    initial_prompt=self.config.initial_prompt or None, vad_filter=self.config.vad_filter,
                    vad_parameters=dict(self.config.vad_parameters), word_timestamps=True,
                    condition_on_previous_text=False, task="transcribe")
                while True:
                    self.check_cancel()
                    # Only inference failures trigger CUDA fallback; disk-write failures must propagate.
                    try:
                        segment = next(iterator)
                    except StopIteration:
                        return
                    watermark = max(done, segments[-1]["end"] if segments else 0)
                    converted = convert_segment(segment, offset, watermark, chunk_end, self.normalizer, segments)
                    if converted:
                        append_segment(partial, converted)
                        segments.append(converted)
                        self.on_segment(converted)
                        self.report(converted["end"], duration)
            except (RuntimeError, ValueError) as exc:
                if self.manager.device != "cuda":
                    raise
                self.logger.warning("CUDA inference failed; replaying current window on CPU", exc_info=True)
                if iterator is not None and hasattr(iterator, "close"):
                    iterator.close()
                iterator = None
                self.warn(f"CUDA 辨識失敗，保留已完成片段並改用 CPU int8：{exc}")
                self.manager.fallback_cpu()
                self._record_device(metadata)
