"""Command line interface with human-readable diagnostics and Rich progress."""
import argparse
from dataclasses import replace
import importlib.metadata
import logging
from pathlib import Path
import sys
import tempfile
import wave

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .audio import discover_inputs, extract_chunk, find_ffmpeg, probe_audio
from .config import Config, load_config
from .errors import TranscriberError
from .formatter import TextNormalizer, elapsed_label
from .models import ModelManager
from .transcriber import Transcriber
from .utils import configure_logging, output_lock


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="本機會議錄音轉錄工具：錄音與逐字稿僅於本機處理，無需 API Key。")
    result.add_argument("input", nargs="?", help="錄音檔案或目錄，預設使用 config.yaml 的 input_dir")
    result.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config.yaml")
    result.add_argument("--model", help="small / medium / large-v3，或本機模型目錄")
    result.add_argument("--language", help="預設 zh")
    result.add_argument("--hotwords", help="本次錄音可能出現的術語；只做提示，不會覆寫原文")
    result.add_argument("--context", action="store_true", default=None,
                        help="啟用區段內連貫上下文（實驗性，可能引起重複文字）")
    result.add_argument("--device", choices=["auto", "cpu", "cuda"])
    result.add_argument("--compute-type")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--offline", action="store_true", default=None, help="禁止下載模型")
    result.add_argument("--diarize", action="store_true", default=None, help="在本機區分不同說話者")
    result.add_argument("--num-speakers", type=int, help="說話者人數，0 代表自動估計")
    result.add_argument("--force", action="store_true", help="先備份舊結果再重新轉錄")
    result.add_argument("--recursive", action="store_true", help="處理目錄下所有子目錄")
    result.add_argument("--self-test", action="store_true", help="離線檢查套件、音訊處理、中文檔名，不下載模型")
    result.add_argument("--download-model", action="store_true", help="只下載或確認指定模型快取")
    return result


def self_test(config: Config, console: Console) -> None:
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise TranscriberError("請使用 64-bit Python 3.11 或 3.12。")
    import av
    import ctranslate2
    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions
    from PySide6 import QtCore
    VadOptions(**config.vad_parameters)
    for package in ("faster-whisper", "ctranslate2", "av", "opencc-python-reimplemented", "imageio-ffmpeg"):
        console.print(f"[OK] {package} {importlib.metadata.version(package)}")
    executable = find_ffmpeg(config.ffmpeg_path)
    console.print(f"[OK] FFmpeg：{executable}")
    normalizer = TextNormalizer(True, {"iq cube": "iCube"})
    if normalizer.normalize("软件 iq cube API") != "軟體 iCube API":
        raise TranscriberError("繁體中文轉換檢查失敗。")
    with tempfile.TemporaryDirectory(prefix="轉錄自我測試-", dir=config.output_dir) as temp:
        source, chunk = Path(temp) / "中文錄音.wav", Path(temp) / "音訊分段.wav"
        with wave.open(str(source), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 32000)
        info = probe_audio(source)
        actual = extract_chunk(executable, source, chunk, 0.5, 1, info.stream_index)
        if abs(info.duration - 2) > 0.01 or abs(actual - 1) > 0.01:
            raise TranscriberError("音訊分段長度檢查失敗。")
    console.print("[OK] 中文路徑、FFmpeg 解碼、音訊分段、繁體轉換")
    console.print(f"[OK] 桌面介面 Qt {QtCore.qVersion()}")
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception:
        count = 0
    console.print(f"[OK] CTranslate2 CUDA devices: {count}（實際載入或推論失敗會回退 CPU）")
    console.print("Self-test passed. 此檢查不下載模型、不執行語音辨識。")


def main(argv: list[str] | None = None) -> int:
    # Redirected Windows output otherwise uses the legacy code page (e.g. cp950).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    console = Console(markup=False, highlight=False)
    log_path = None
    try:
        config = load_config(args.config, {"model": args.model, "language": args.language,
                             "device": args.device, "compute_type": args.compute_type, "offline": args.offline,
                             "diarize": args.diarize, "num_speakers": args.num_speakers,
                             "hotwords": args.hotwords, "condition_on_previous_text": args.context})
        if args.output_dir:
            config = replace(config, output_dir=args.output_dir.resolve())
        for folder in (config.input_dir, config.output_dir, config.models_dir, config.logs_dir):
            folder.mkdir(parents=True, exist_ok=True)
        log_path = configure_logging(config.logs_dir)
        console.print("本機會議錄音轉錄工具\n所有錄音與逐字稿均於本機處理；程式不會主動將音訊上傳到第三方服務。")
        warn = lambda message: console.print(f"[WARN] {message}")
        if args.self_test:
            self_test(config, console)
            return 0
        if args.download_model:
            console.print(f"模型快取：{ModelManager(config, warn).resolve_path()}")
            return 0
        source = Path(args.input).resolve() if args.input else config.input_dir
        files = discover_inputs(source, args.recursive, (config.output_dir, config.models_dir, config.logs_dir))
        if not files:
            console.print(f"未找到錄音檔。請將 .m4a / .mp3 / .wav 等錄音放入：{source}")
            return 0
        failures = 0
        with output_lock(config.output_dir):
            worker = Transcriber(config, warn)
            for file in files:
                console.print(f"\n檔案：{file.name}\n模型：{config.model}\n語言：{config.language}")
                with Progress(TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
                              TextColumn("{task.fields[timeline]}"), TimeElapsedColumn(), console=console) as progress:
                    task_id = progress.add_task("準備音訊與模型…", total=1, timeline="")
                    def update(done: float, total: float, device: str) -> None:
                        progress.update(task_id, total=total, completed=min(done, total),
                                        description=f"Transcribing ({device.upper()})",
                                        timeline=f"{elapsed_label(done)} / {elapsed_label(total)}")
                    worker.progress = update
                    try:
                        result = worker.transcribe(file, force=args.force)
                        if result.skipped:
                            progress.update(task_id, visible=False)
                        else:
                            progress.update(task_id, completed=result.duration, total=result.duration)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        failures += 1
                        logging.getLogger(__name__).exception("Failed source=%s", file)
                        console.print(f"[ERROR] {exc}\n已保存的 checkpoint 保留；下次執行會接續。Log：{log_path}")
                        continue
                if result.skipped:
                    console.print(f"[SKIP] {file.name} 已完成轉錄。\nOutput: {result.folder}")
                else:
                    console.print(f"\n=================================\nTranscription completed\n"
                                  f"File: {file.name}\nDuration: {elapsed_label(result.duration)}\n"
                                  f"Processing: {elapsed_label(result.processing_seconds)}\nOutput: {result.folder}\nGenerated:")
                    for name in result.generated:
                        console.print(f"✓ {name}")
                    console.print("=================================")
        return 1 if failures else 0
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user", exc_info=True)
        console.print("\n[WARN] 已中斷；已寫入的片段保留，下次執行會接續。")
        return 130
    except Exception as exc:
        if log_path:
            logging.getLogger(__name__).exception("CLI failed")
        console.print(f"[ERROR] {exc}")
        if log_path:
            console.print(f"詳細錯誤記錄：{log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
