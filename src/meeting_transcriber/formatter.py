"""Safe text normalization and UTF-8 transcript exporters."""
from dataclasses import asdict
import math
from pathlib import Path
import re
from typing import Any

from .config import Config, OutputConfig
from .utils import atomic_write, write_json


def timestamp(seconds: float, srt: bool = False) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Timestamp must be finite and nonnegative")
    total = int(math.floor(seconds * 1000 + 0.5)) if srt else int(seconds) * 1000
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{base},{millis:03d}" if srt else base


def elapsed_label(seconds: float) -> str:
    minutes, secs = divmod(int(max(0, seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


class TextNormalizer:
    def __init__(self, traditional: bool, terminology: dict[str, str]):
        self.converter = None
        if traditional:
            from opencc import OpenCC
            self.converter = OpenCC("s2twp")
        self.terms = {k.casefold(): v for k, v in terminology.items()}
        patterns = []
        for term in sorted(self.terms, key=len, reverse=True):
            left = r"(?<![A-Za-z0-9_])" if term[0].isascii() and term[0].isalnum() else ""
            right = r"(?![A-Za-z0-9_])" if term[-1].isascii() and term[-1].isalnum() else ""
            patterns.append(left + re.escape(term) + right)
        self.pattern = re.compile("|".join(patterns), re.IGNORECASE) if patterns else None

    def normalize(self, text: str) -> str:
        if self.converter:
            text = self.converter.convert(text)
        if self.pattern:
            text = self.pattern.sub(lambda match: self.terms[match.group(0).casefold()], text)
        return text.strip()


def suspicious_reasons(text: str, previous: list[dict[str, Any]], *,
                       compression_ratio: float = 0, no_speech_prob: float = 0,
                       avg_logprob: float = 0) -> list[str]:
    reasons = []
    cleaned = re.sub(r"[\W_]", "", text).casefold()
    repeated = cleaned and sum(re.sub(r"[\W_]", "", s["text"]).casefold() == cleaned for s in previous[-5:]) >= 2
    if repeated:
        reasons.append("repeated_text")
    phrases = ("謝謝觀看", "感謝收看", "請訂閱", "字幕提供", "字幕由")
    if any(phrase in text for phrase in phrases) and (repeated or no_speech_prob > 0.6):
        reasons.append("possible_outro_hallucination")
    if compression_ratio > 2.4:
        reasons.append("high_compression_ratio")
    if no_speech_prob > 0.6 and avg_logprob < -1:
        reasons.append("low_confidence_speech")
    return reasons


def render_srt(segments: list[dict[str, Any]]) -> str:
    cues = []
    for index, segment in enumerate(segments, 1):
        # Avoid blank lines within a cue, which would terminate it early.
        text = "\n".join(line.strip() for line in display_text(segment).splitlines() if line.strip())
        start_ms = int(math.floor(segment["start"] * 1000 + 0.5))
        end_ms = max(start_ms + 1, int(math.floor(segment["end"] * 1000 + 0.5)))
        cues.append(f"{index}\n{timestamp(start_ms / 1000, True)} --> {timestamp(end_ms / 1000, True)}\n{text}\n")
    return "\n".join(cues)


AI_PROMPT = """---

## AI 整理提示

請依據以上逐字稿整理正式會議記錄。

規則：

1. 不得自行新增逐字稿未提及的決議。
2. 將內容分類為：現況、問題、需求、決議、待確認事項、Action Items。
3. 專有名詞：iCube、Wafer、Waffle、Barcode、ERP、SMT。
4. 使用「批次／批號」，不得寫成 P 次。
5. 如果內容無法確定，標示「待確認」，不要猜測。
"""


def render_markdown(name: str, metadata: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    language = "中文" if metadata["language"] == "zh" else metadata["language"]
    lines = [f"# {name}－逐字稿", "", f"- 檔案：{metadata['file']}",
             f"- 長度：{elapsed_label(metadata['duration'])}", f"- 語言：{language}",
             f"- 模型：{metadata['model']}", "", "---", ""]
    bucket = None
    for segment in segments:
        current = int(segment["start"] // 600)
        if current != bucket:
            bucket = current
            lines += [f"## {elapsed_label(bucket * 600)}–{elapsed_label((bucket + 1) * 600)}", ""]
        lines += [display_text(segment), ""]
    if not segments:
        lines += ["（未辨識到語音內容。）", ""]
    lines.append(AI_PROMPT)
    return "\n".join(lines)


def display_text(segment: dict[str, Any]) -> str:
    speaker = segment.get("speaker")
    if not speaker:
        return segment["text"]
    label = f"說話者 {int(speaker.removeprefix('SPEAKER_'))}" if speaker.startswith("SPEAKER_") else "未能區分"
    return f"【{label}】{segment['text']}"


def expected_outputs(folder: Path, name: str, config: OutputConfig) -> dict[str, Path]:
    # Never overwrite control files with a recording's JSON export.
    if name.casefold() in {"metadata", "checkpoint"}:
        name += ".transcript"
    extensions = {"txt": ".txt", "timestamp_txt": ".timestamps.txt", "srt": ".srt",
                  "json": ".json", "markdown": ".md"}
    return {key: folder / (name + extension) for key, extension in extensions.items() if asdict(config)[key]}


def write_outputs(folder: Path, name: str, metadata: dict[str, Any],
                  segments: list[dict[str, Any]], config: Config) -> list[str]:
    outputs = expected_outputs(folder, name, config.output)
    for kind, path in outputs.items():
        if kind == "json":
            write_json(path, {**metadata, "segments": segments})
            continue
        if kind == "txt":
            text = "\n\n".join(display_text(s) for s in segments) + ("\n" if segments else "")
        elif kind == "timestamp_txt":
            text = "\n\n".join(f"[{timestamp(s['start'])} - {timestamp(s['end'])}]\n{display_text(s)}" for s in segments)
            text += "\n" if segments else ""
        elif kind == "srt":
            text = render_srt(segments)
        else:
            text = render_markdown(name, metadata, segments)
        atomic_write(path, text)
    return [p.name for p in outputs.values()]
