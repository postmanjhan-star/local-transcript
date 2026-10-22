"""Validated YAML configuration; relative paths are relative to the YAML file."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError


@dataclass(frozen=True)
class OutputConfig:
    txt: bool = True
    timestamp_txt: bool = True
    srt: bool = True
    json: bool = True
    markdown: bool = True


@dataclass(frozen=True)
class Config:
    model: str = "medium"
    language: str = "zh"
    device: str = "auto"
    compute_type: str = "auto"
    traditional_chinese: bool = True
    beam_size: int = 5
    vad_filter: bool = True
    vad_parameters: dict[str, Any] = field(default_factory=dict)
    chunk_seconds: float = 300
    overlap_seconds: float = 15
    cpu_threads: int = 0
    offline: bool = False
    diarize: bool = False
    num_speakers: int = 0
    speaker_threshold: float = 0.45
    output: OutputConfig = field(default_factory=OutputConfig)
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    ffmpeg_path: str | None = None
    initial_prompt: str = ""
    hotwords: str = ""
    condition_on_previous_text: bool = False
    terminology: dict[str, str] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Only recognition/normalization settings affect resumability."""
        data = asdict(self)
        for key in ("device", "compute_type", "cpu_threads", "offline", "output",
                    "input_dir", "output_dir", "models_dir", "logs_dir", "ffmpeg_path",
                    "diarize", "num_speakers", "speaker_threshold"):
            data.pop(key)
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _number(value: Any, name: str, low: float, high: float, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} 必須是數值。")
    if not math.isfinite(value) or not low <= value <= high or (integer and not isinstance(value, int)):
        raise ConfigError(f"{name} 必須介於 {low} 與 {high} 之間" + ("且為整數。" if integer else "。"))


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> Config:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if data is None:
            data = {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"無法讀取設定檔 {path}：{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config.yaml 最上層必須為 key: value。")
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = set(data) - set(Config.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"不認識的設定：{', '.join(sorted(map(str, unknown)))}")
    output = data.pop("output", {})
    if not isinstance(output, dict) or set(output) - set(OutputConfig.__dataclass_fields__):
        raise ConfigError("output 包含不支援的格式設定。")
    if any(type(v) is not bool for v in output.values()):
        raise ConfigError("output 格式開關必須是 true / false。")
    data["output"] = OutputConfig(**output)
    for key in ("input_dir", "output_dir", "models_dir", "logs_dir"):
        value = data.get(key, key.removesuffix("_dir"))
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} 必須是有效路徑。")
        p = Path(value).expanduser()
        data[key] = (path.resolve().parent / p).resolve()
    cfg = Config(**data)
    for key in ("model", "language", "device", "compute_type"):
        if not isinstance(getattr(cfg, key), str) or not getattr(cfg, key).strip():
            raise ConfigError(f"{key} 必須為非空字串。")
    if cfg.device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("device 只接受 auto / cpu / cuda。")
    if cfg.compute_type not in {"auto", "int8", "int8_float16", "int8_float32", "int8_bfloat16",
                                "float16", "float32", "bfloat16", "int16", "default"}:
        raise ConfigError("compute_type 不受支援。")
    for key in ("traditional_chinese", "vad_filter", "offline", "diarize", "condition_on_previous_text"):
        if type(getattr(cfg, key)) is not bool:
            raise ConfigError(f"{key} 必須是 true / false。")
    _number(cfg.beam_size, "beam_size", 1, 20, True)
    _number(cfg.cpu_threads, "cpu_threads", 0, 256, True)
    _number(cfg.num_speakers, "num_speakers", 0, 20, True)
    _number(cfg.speaker_threshold, "speaker_threshold", 0.05, 1.5)
    _number(cfg.chunk_seconds, "chunk_seconds", 30, 900)
    _number(cfg.overlap_seconds, "overlap_seconds", 0, min(30, cfg.chunk_seconds / 2))
    if not isinstance(cfg.initial_prompt, str):
        raise ConfigError("initial_prompt 必須為字串。")
    if not isinstance(cfg.hotwords, str):
        raise ConfigError("hotwords 必須為字串。")
    if not isinstance(cfg.terminology, dict) or any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
        for k, v in cfg.terminology.items()
    ):
        raise ConfigError("terminology 必須為非空字串對照表。")
    normalized: dict[str, str] = {}
    for k, v in cfg.terminology.items():
        if k.casefold() in normalized and normalized[k.casefold()] != v:
            raise ConfigError(f"terminology 的大小寫規則互相衝突：{k}")
        normalized[k.casefold()] = v
    if not isinstance(cfg.vad_parameters, dict):
        raise ConfigError("vad_parameters 必須為對照表。")
    limits = {"threshold": (0, 1, False), "neg_threshold": (0, 1, False),
              "min_speech_duration_ms": (0, 60000, True), "max_speech_duration_s": (1, 900, False),
              "min_silence_duration_ms": (0, 60000, True), "speech_pad_ms": (0, 5000, True)}
    for key, value in cfg.vad_parameters.items():
        if key not in limits:
            raise ConfigError(f"不支援的 VAD 參數：{key}")
        _number(value, f"vad_parameters.{key}", *limits[key])
    if cfg.ffmpeg_path is not None and (not isinstance(cfg.ffmpeg_path, str) or not cfg.ffmpeg_path.strip()):
        raise ConfigError("ffmpeg_path 必須為路徑字串或 null。")
    # Explicit relative executable/model paths are resolved against the config.
    from dataclasses import replace
    updates: dict[str, Any] = {}
    if cfg.ffmpeg_path and not Path(cfg.ffmpeg_path).is_absolute():
        updates["ffmpeg_path"] = str((path.parent / cfg.ffmpeg_path).resolve())
    candidate = (path.parent / cfg.model).resolve()
    if candidate.is_dir():
        updates["model"] = str(candidate)
    return replace(cfg, **updates)
