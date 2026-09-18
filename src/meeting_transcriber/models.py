"""Local model cache resolution and verified CTranslate2 device fallback."""
import gc
import logging
import os
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .errors import ModelDownloadError, ModelLoadError


class ModelManager:
    def __init__(self, config: Config, warn: Callable[[str], None]):
        self.config = config
        self.warn = warn
        self.model: Any = None
        self.path: str | None = None
        self.device = "uninitialized"
        self.compute_type = "uninitialized"
        self.logger = logging.getLogger(__name__)

    def resolve_path(self) -> str:
        if self.path:
            return self.path
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from faster_whisper.utils import download_model
        source = Path(self.config.model)
        if source.is_dir():
            candidate = str(source.resolve())
        else:
            self.config.models_dir.mkdir(parents=True, exist_ok=True)
            try:
                # Cache first: completed model runs make no network requests.
                candidate = download_model(self.config.model, cache_dir=str(self.config.models_dir), local_files_only=True)
                self._validate_model(candidate)
            except Exception as exc:
                if self.config.offline:
                    raise ModelDownloadError(f"離線模式找不到完整 Whisper 模型：{self.config.model}。請先連網下載。") from exc
                self.warn(f"本機尚無完整 {self.config.model} 模型，開始下載（僅下載模型，不上傳音訊）。")
                try:
                    candidate = download_model(self.config.model, cache_dir=str(self.config.models_dir))
                except Exception as download_exc:
                    raise ModelDownloadError("Whisper 模型下載失敗。請檢查網路、磁碟空間或改用本機模型路徑。") from download_exc
        self._validate_model(candidate)
        self.path = candidate
        return candidate

    @staticmethod
    def _validate_model(path: str) -> None:
        for name in ("model.bin", "config.json", "tokenizer.json"):
            if not (Path(path) / name).is_file():
                raise ModelLoadError(f"模型不完整，缺少 {name}：{path}")

    def ensure_loaded(self) -> Any:
        if self.model is not None:
            return self.model
        model_path = self.resolve_path()
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except (ImportError, OSError) as exc:
            raise ModelLoadError("無法載入 faster-whisper / CTranslate2，請重新執行 setup.ps1。") from exc
        cuda = False
        if self.config.device != "cpu":
            try:
                cuda = ctranslate2.get_cuda_device_count() > 0
                if cuda:
                    supported = ctranslate2.get_supported_compute_types("cuda")
                    desired = "float16" if self.config.compute_type == "auto" else self.config.compute_type
                    if desired not in supported and desired != "default":
                        raise RuntimeError(f"CUDA does not support {desired}")
                    self.model = WhisperModel(model_path, device="cuda", compute_type=desired,
                                              cpu_threads=self.config.cpu_threads, local_files_only=True)
                    self.device, self.compute_type = "cuda", desired
                    return self.model
            except Exception:
                self.logger.warning("CUDA initialization failed", exc_info=True)
                self.warn("CUDA 無法初始化，改用 CPU int8 模式。")
                return self.fallback_cpu()
            if not cuda:
                self.warn("CUDA 不可用，使用 CPU int8。")
        desired_cpu = self.config.compute_type if self.config.device == "cpu" and self.config.compute_type != "auto" else "int8"
        return self._load_cpu(desired_cpu)

    def _load_cpu(self, compute: str) -> Any:
        from faster_whisper import WhisperModel
        try:
            self.model = WhisperModel(self.resolve_path(), device="cpu", compute_type=compute,
                                      cpu_threads=self.config.cpu_threads, local_files_only=True)
        except Exception as exc:
            raise ModelLoadError(f"CPU 模型載入失敗：{exc}") from exc
        self.device, self.compute_type = "cpu", compute
        return self.model

    def fallback_cpu(self) -> Any:
        self.model = None
        gc.collect()
        return self._load_cpu("int8")

