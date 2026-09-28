"""Local speaker embeddings, global clustering, and word-level speaker labels.

Audio is decoded in bounded chunks. Only compact embeddings span the recording.
No speaker names or identities are inferred; labels are local to one recording.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import urllib.request
import wave

import numpy as np

from .audio import extract_chunk, find_ffmpeg, probe_audio
from .config import Config
from .errors import CancelledError, DiarizationError
from .formatter import TextNormalizer
from .progress import ProgressEvent
from .utils import read_json, source_identity, write_json

MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
MODEL_SHA256 = "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b"
DIARIZATION_VERSION = 3
MIN_REFERENCE_SECONDS = 1.5


def speaker_vad_parameters(config: Config) -> dict[str, Any]:
    # ASR benefits from generous silence/padding; speaker embeddings must avoid
    # joining adjacent turns. Leave the transcription VAD configuration intact.
    return {**config.vad_parameters,
            "min_silence_duration_ms": min(config.vad_parameters.get("min_silence_duration_ms", 200), 200),
            "speech_pad_ms": min(config.vad_parameters.get("speech_pad_ms", 60), 60)}


def speech_windows(segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
    intervals = []
    for segment in segments:
        for unit in segment.get("words") or [segment]:
            if unit["end"] > unit["start"]:
                intervals.append((float(unit["start"]), float(unit["end"])))
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    windows = []
    for start, end in merged:
        duration = end - start
        if duration < 0.5:
            continue
        # Keep the final window full-length instead of creating a less reliable
        # short tail of the same utterance. Never bridge a detected silence.
        count = max(1, int(np.ceil((duration - 3) / 1.5)) + 1)
        for offset in np.linspace(start, max(start, end - 3), count):
            windows.append((round(float(offset), 6), round(min(end, offset + 3), 6)))
    return windows


def cluster_embeddings(embeddings: np.ndarray, num_speakers: int, threshold: float,
                       durations: np.ndarray | None = None) -> list[int]:
    """Return stable local labels; -1 means insufficient assignment evidence."""
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) == 0:
        return []
    if len(embeddings) == 1 or num_speakers == 1:
        return [0] * len(embeddings)
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    # Bound the quadratic clustering matrix for two-hour recordings and beyond.
    references = np.arange(len(normalized))
    if not num_speakers and durations is not None:
        reliable = references[np.asarray(durations) >= MIN_REFERENCE_SECONDS]
        if len(reliable):
            references = reliable
    selection = references[np.unique(np.linspace(0, len(references) - 1, min(2000, len(references)), dtype=int))]
    sample = normalized[selection]
    count = min(num_speakers, len(sample)) if num_speakers else None
    clustering = AgglomerativeClustering(n_clusters=count, metric="cosine", linkage="average",
                                         distance_threshold=None if count else threshold)
    sample_labels = clustering.fit_predict(sample) if len(sample) > 1 else np.zeros(1, dtype=int)

    def centers(labels: np.ndarray) -> np.ndarray:
        values = np.stack([sample[labels == label].mean(axis=0) for label in sorted(set(labels))])
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)

    centroids = centers(sample_labels)
    if not num_speakers and len(centroids) > 1:
        # Within-speaker variation can inflate average pairwise distance. A
        # stricter second pass compares pooled voices; complete linkage avoids
        # merging a chain of individually similar but distinct voices.
        merged = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete",
                                         distance_threshold=threshold * 0.8).fit_predict(centroids)
        sample_labels = merged[sample_labels]
        centroids = centers(sample_labels)
        # centers() sorts labels, so compact them before restoring sample labels.
        sample_labels = np.searchsorted(sorted(set(sample_labels)), sample_labels)
    labels = np.empty(len(normalized), dtype=int)
    # Bound assignment memory too, independently of recording duration.
    for start in range(0, len(normalized), 512):
        distances = 1 - normalized[start:start + 512] @ centroids.T
        best = np.argmin(distances, axis=1)
        labels[start:start + len(best)] = best
        if not num_speakers:
            uncertain = distances[np.arange(len(best)), best] > threshold
            if len(centroids) > 1:
                closest = np.partition(distances, 1, axis=1)[:, :2]
                uncertain |= closest[:, 1] - closest[:, 0] < 0.08
            best[uncertain] = -1
            labels[start:start + len(best)] = best
    # Reference windows retain the actual clustering result. Confidence gating
    # applies to short speech and unsampled windows, not forced speaker counts.
    labels[selection] = sample_labels
    # Stable numbering follows first appearance, not arbitrary cluster indices.
    names: dict[int, int] = {}
    return [names.setdefault(int(label), len(names)) if label >= 0 else -1 for label in labels]


def label_segments(segments: list[dict[str, Any]], windows: list[dict[str, Any]],
                   config: Config) -> list[dict[str, Any]]:
    normalizer = TextNormalizer(config.traditional_chinese, config.terminology)
    windows = sorted(windows, key=lambda w: w["start"])
    starts = np.array([w["start"] for w in windows])
    max_duration = max((w["end"] - w["start"] for w in windows), default=0)

    def speaker_at(start: float, end: float) -> str:
        if not windows:
            return "UNKNOWN"
        # Vote by actual overlap, emphasizing the center of each embedding
        # window. A nearby center must not label words outside that window.
        if end <= start:
            start, end = start - 0.01, start + 0.01
        first = int(np.searchsorted(starts, start - max_duration, side="left"))
        last = int(np.searchsorted(starts, end, side="right"))
        votes: dict[int, float] = {}
        for window in windows[first:last]:
            left, right = max(start, window["start"]), min(end, window["end"])
            if right <= left:
                continue
            width = window["end"] - window["start"]
            # Integral of a triangular center weight over the word interval.
            def area(t: float) -> float:
                x = (t - window["start"]) / width
                return width * (x * x if x <= 0.5 else 0.5 - (1 - x) ** 2)
            label = window["speaker"]
            votes[label] = votes.get(label, 0) + area(right) - area(left)
        if not votes:
            return "UNKNOWN"
        ranked = sorted(votes, key=votes.get, reverse=True)
        label = ranked[0]
        if label < 0 or (len(ranked) > 1 and votes[label] - votes[ranked[1]] <= 0.1 * sum(votes.values())):
            return "UNKNOWN"
        return f"SPEAKER_{label + 1:02d}"

    result = []
    for segment in segments:
        words = segment.get("words") or []
        # Older checkpoints have no word alignment. Keep their text intact.
        if not words or "".join(w["word"] for w in words).strip() != segment.get("raw_text", segment["text"]).strip():
            result.append({**segment, "id": len(result), "speaker": speaker_at(segment["start"], segment["end"]),
                           "speaker_alignment": "segment"})
            continue
        assignments = [speaker_at(w["start"], w["end"]) for w in words]
        groups: list[tuple[str, list[dict[str, Any]]]] = []
        for word, speaker in zip(words, assignments):
            if groups and groups[-1][0] == speaker:
                groups[-1][1].append(word)
            else:
                groups.append((speaker, [word]))
        for speaker, group in groups:
            raw = "".join(w["word"] for w in group)
            start, end = group[0]["start"], group[-1]["end"]
            if end <= start:
                # Zero-duration punctuation must remain in the transcript.
                if result and result[-1].get("source_segment_id") == segment["id"]:
                    result[-1]["raw_text"] += raw
                    result[-1]["text"] = normalizer.normalize(result[-1]["raw_text"])
                    result[-1]["words"].extend(group)
                    continue
                start, end = segment["start"], segment["end"]
            result.append({**segment, "id": len(result), "source_segment_id": segment["id"],
                           "start": start, "end": end, "raw_text": raw,
                           "text": normalizer.normalize(raw), "words": group, "speaker": speaker,
                           "speaker_alignment": "word"})
    return result


class Diarizer:
    def __init__(self, config: Config, warn: Callable[[str], None],
                 progress: Callable[[ProgressEvent], None], cancelled: Callable[[], bool]):
        self.config, self.warn, self.progress, self.cancelled = config, warn, progress, cancelled

    def check_cancel(self) -> None:
        if self.cancelled():
            raise CancelledError("已停止，已完成的逐字稿及說話者分析進度已保留。")

    def model_path(self) -> Path:
        target = self.config.models_dir / "speakers" / "3dspeaker.onnx"
        if target.is_file():
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA256:
                    return target
        if self.config.offline:
            raise DiarizationError("離線模式找不到完整說話者模型。請取消離線模式，首次下載後即可離線使用。")
        self.warn("首次使用說話者區分，下載約 40 MB 本機模型；不會上傳音訊。")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".download")
        try:
            request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "meeting-transcriber/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as stream:
                total = int(response.headers.get("Content-Length", 0))
                done = 0
                while block := response.read(1024 * 1024):
                    self.check_cancel()
                    stream.write(block)
                    done += len(block)
                    self.progress(ProgressEvent("speaker_download", done, total))
                stream.flush()
                os.fsync(stream.fileno())
            with temporary.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != MODEL_SHA256:
                raise DiarizationError("說話者模型校驗失敗，請重新執行下載。")
            os.replace(temporary, target)
        except CancelledError:
            raise
        except Exception as exc:
            raise DiarizationError(f"說話者模型下載失敗：{exc}") from exc
        return target

    def run(self, source: Path, folder: Path, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .speaker_process import run_diarization
        return run_diarization(self.config, source, folder, segments, self.warn, self.progress, self.cancelled)

    def _run_local(self, source: Path, folder: Path, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.check_cancel()
        if not segments:
            return label_segments(segments, [], self.config)
        if any(not segment.get("words") for segment in segments):
            self.warn("部分舊逐字稿沒有逐字時間戳，會按原片段標示說話者；重新轉錄可改善切換位置。")
        # Detect speech independently of the transcript: ASR can omit words or
        # misalign timestamps, and must not determine the speaker population.
        vad_parameters = speaker_vad_parameters(self.config)
        signature = hashlib.sha256(json.dumps([DIARIZATION_VERSION, MODEL_SHA256, source_identity(source),
                                   vad_parameters], sort_keys=True).encode()).hexdigest()
        cache = folder / "speaker-checkpoints" / signature[:16]
        cache.mkdir(parents=True, exist_ok=True)
        info = probe_audio(source)
        batches = list(range(int(np.ceil(info.duration / 300))))
        extractor = None
        runtime = None  # Keep the explicitly loaded Windows DLL alive for the extractor.
        ffmpeg = None
        records = []
        completed = 0.0
        saved: dict[int, list[dict[str, Any]]] = {}
        for batch in batches:
            path = cache / f"{batch}.json"
            if path.exists():
                data = read_json(path)
                if data.get("signature") == signature:
                    saved[batch] = data["records"]
                    completed += min(300, info.duration - batch * 300)
        self.progress(ProgressEvent("diarizing", completed, info.duration, "cpu"))
        with tempfile.TemporaryDirectory(prefix="speaker-audio-", dir=folder) as temp:
            for batch in batches:
                self.check_cancel()
                if batch in saved:
                    records.extend(saved[batch])
                    continue
                if ffmpeg is None:
                    if os.name == "nt":
                        # System32 can contain ORT 1.17; Sherpa requires a newer API.
                        # Load only our uv-installed runtime, never replace system DLLs.
                        import ctypes
                        import onnxruntime
                        dll = Path(onnxruntime.__file__).parent / "capi" / "onnxruntime.dll"
                        runtime = ctypes.WinDLL(str(dll))
                    ffmpeg = find_ffmpeg(self.config.ffmpeg_path)
                start = batch * 300
                end = min(info.duration, start + 300)
                chunk = Path(temp) / "chunk.wav"
                extract_chunk(ffmpeg, source, chunk, start, end - start, info.stream_index)
                with wave.open(str(chunk), "rb") as audio:
                    samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
                from faster_whisper.vad import VadOptions, get_speech_timestamps
                intervals = get_speech_timestamps(samples, VadOptions(**vad_parameters))
                planned = speech_windows([{"start": start + v["start"] / 16000,
                                           "end": start + v["end"] / 16000} for v in intervals])
                if planned and extractor is None:
                    import sherpa_onnx
                    path = self.model_path()
                    self.check_cancel()
                    options = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=str(path), num_threads=self.config.cpu_threads or 2, provider="cpu")
                    if not options.validate():
                        raise DiarizationError("說話者模型無法載入。")
                    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(options)
                    self.progress(ProgressEvent("diarizing", completed, info.duration, "cpu"))
                current = []
                for a, b in planned:
                    self.check_cancel()
                    waveform = np.ascontiguousarray(samples[max(0, round((a - start) * 16000)):round((b - start) * 16000)])
                    stream = extractor.create_stream()
                    stream.accept_waveform(sample_rate=16000, waveform=waveform)
                    stream.input_finished()
                    if extractor.is_ready(stream):
                        vector = np.asarray(extractor.compute(stream), dtype=np.float32)
                        if np.isfinite(vector).all() and np.linalg.norm(vector) > 1e-9:
                            current.append({"start": a, "end": b, "embedding": vector.tolist()})
                    self.progress(ProgressEvent("diarizing", completed + b - start, info.duration, "cpu"))
                completed += end - start
                write_json(cache / f"{batch}.json", {"signature": signature, "records": current})
                records.extend(current)
                self.progress(ProgressEvent("diarizing", completed, info.duration, "cpu"))
        self.check_cancel()
        self.progress(ProgressEvent("clustering"))
        records.sort(key=lambda row: row["start"])
        vectors = np.array([row["embedding"] for row in records], dtype=np.float32)
        durations = np.array([row["end"] - row["start"] for row in records])
        if len(durations) and not self.config.num_speakers and not np.any(durations >= MIN_REFERENCE_SECONDS):
            self.warn("錄音只有短語音片段，自動人數較不可靠；已知人數時可指定人數再分析。")
        labels = cluster_embeddings(vectors, self.config.num_speakers, self.config.speaker_threshold, durations)
        labelled_windows = [{"start": row["start"], "end": row["end"], "speaker": label}
                            for row, label in zip(records, labels)]
        self.check_cancel()
        result = label_segments(segments, labelled_windows, self.config)
        write_json(folder / "diarization.json", {"model": "3D-Speaker ERes2Net Chinese", "model_sha256": MODEL_SHA256,
                   "diarization_version": DIARIZATION_VERSION,
                   "num_speakers_requested": self.config.num_speakers, "windows": labelled_windows,
                   "speakers": sorted({s["speaker"] for s in result}), "segments": result})
        return result
