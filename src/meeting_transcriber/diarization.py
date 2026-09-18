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


def speech_windows(segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
    intervals = []
    for segment in segments:
        for unit in segment.get("words") or [segment]:
            if unit["end"] > unit["start"]:
                intervals.append((float(unit["start"]), float(unit["end"])))
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 0.8:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    windows = []
    for start, end in merged:
        while end - start >= 0.5:
            finish = min(end, start + 3)
            windows.append((round(start, 6), round(finish, 6)))
            if finish >= end:
                break
            start += 1.5
    return windows


def cluster_embeddings(embeddings: np.ndarray, num_speakers: int, threshold: float) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) == 0:
        return []
    if len(embeddings) == 1 or num_speakers == 1:
        return [0] * len(embeddings)
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    # Bound the quadratic clustering matrix for two-hour recordings and beyond.
    selection = np.unique(np.linspace(0, len(normalized) - 1, min(2000, len(normalized)), dtype=int))
    sample = normalized[selection]
    count = min(num_speakers, len(sample)) if num_speakers else None
    clustering = AgglomerativeClustering(n_clusters=count, metric="cosine", linkage="average",
                                         distance_threshold=None if count else threshold)
    sample_labels = clustering.fit_predict(sample)
    if len(selection) == len(normalized):
        labels = sample_labels.tolist()
    else:
        centroids = np.stack([sample[sample_labels == label].mean(axis=0) for label in sorted(set(sample_labels))])
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
        labels = np.argmax(normalized @ centroids.T, axis=1).tolist()
    # Stable numbering follows first appearance, not arbitrary cluster indices.
    names: dict[int, int] = {}
    return [names.setdefault(int(label), len(names)) for label in labels]


def label_segments(segments: list[dict[str, Any]], windows: list[dict[str, Any]],
                   config: Config) -> list[dict[str, Any]]:
    normalizer = TextNormalizer(config.traditional_chinese, config.terminology)
    centers = np.array([(w["start"] + w["end"]) / 2 for w in windows])

    def speaker_at(start: float, end: float) -> str:
        if not windows:
            return "UNKNOWN"
        midpoint = (start + end) / 2
        idx = int(np.argmin(np.abs(centers - midpoint)))
        window = windows[idx]
        if midpoint < window["start"] - 1 or midpoint > window["end"] + 1:
            return "UNKNOWN"
        return f"SPEAKER_{window['speaker'] + 1:02d}"

    result = []
    for segment in segments:
        words = segment.get("words") or []
        # Older checkpoints have no word alignment. Keep their text intact.
        if not words or "".join(w["word"] for w in words).strip() != segment.get("raw_text", segment["text"]).strip():
            result.append({**segment, "id": len(result), "speaker": speaker_at(segment["start"], segment["end"]),
                           "speaker_alignment": "segment"})
            continue
        assignments = [speaker_at(w["start"], w["end"]) for w in words]
        for i in range(1, len(words) - 1):
            if assignments[i - 1] == assignments[i + 1] and words[i]["end"] - words[i]["start"] < 0.7:
                assignments[i] = assignments[i - 1]
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
        signature = hashlib.sha256(json.dumps([2, MODEL_SHA256, source_identity(source),
                                   self.config.vad_parameters], sort_keys=True).encode()).hexdigest()
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
                intervals = get_speech_timestamps(samples, VadOptions(**self.config.vad_parameters))
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
        labels = cluster_embeddings(vectors, self.config.num_speakers, self.config.speaker_threshold)
        labelled_windows = [{"start": row["start"], "end": row["end"], "speaker": label}
                            for row, label in zip(records, labels)]
        self.check_cancel()
        result = label_segments(segments, labelled_windows, self.config)
        write_json(folder / "diarization.json", {"model": "3D-Speaker ERes2Net Chinese", "model_sha256": MODEL_SHA256,
                   "num_speakers_requested": self.config.num_speakers, "windows": labelled_windows,
                   "speakers": sorted({s["speaker"] for s in result}), "segments": result})
        return result
