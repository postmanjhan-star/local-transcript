"""Isolate Sherpa's native ONNX runtime from faster-whisper's ONNX runtime."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

from .config import Config
from .errors import CancelledError, DiarizationError
from .progress import ProgressEvent
from .utils import read_json, write_json


def run_diarization(config: Config, source: Path, folder: Path, segments: list[dict[str, Any]],
                    warn: Callable[[str], None], progress: Callable[[ProgressEvent], None],
                    cancelled: Callable[[], bool]) -> list[dict[str, Any]]:
    if cancelled():
        raise CancelledError("已停止說話者分析。")
    with tempfile.TemporaryDirectory(prefix="speaker-job-", dir=folder) as name:
        job = Path(name)
        values = asdict(config)
        for key in ("input_dir", "output_dir", "models_dir", "logs_dir"):
            values[key] = str(values[key])
        write_json(job / "request.json", {"config": values, "source": str(source), "folder": str(folder), "segments": segments})
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get("PYTHONPATH", "")
        messages: queue.Queue[str] = queue.Queue()
        with (job / "stderr.log").open("w", encoding="utf-8") as errors:
            process = subprocess.Popen([sys.executable, "-m", "meeting_transcriber.speaker_worker", str(job)],
                stdout=subprocess.PIPE, stderr=errors, stdin=subprocess.DEVNULL, env=environment,
                text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

            def read_output() -> None:
                for line in process.stdout:
                    messages.put(line)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            requested_at = None
            try:
                while process.poll() is None or reader.is_alive() or not messages.empty():
                    if cancelled():
                        if requested_at is None:
                            (job / "cancel").touch()
                            requested_at = time.monotonic()
                        elif time.monotonic() - requested_at > 5 and process.poll() is None:
                            process.terminate()
                    try:
                        line = messages.get(timeout=.1)
                    except queue.Empty:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("type") == "progress":
                            progress(ProgressEvent(**event["event"]))
                        elif event.get("type") == "message":
                            warn(event["text"])
                    except (ValueError, TypeError):
                        continue  # Native libraries may also write diagnostics to stdout.
                process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                reader.join(timeout=2)
                process.stdout.close()
        if cancelled():
            raise CancelledError("已停止說話者分析，原始逐字稿與已完成分析進度保留。")
        response = job / "result.json"
        if process.returncode != 0 or not response.exists():
            import logging
            detail = (job / "stderr.log").read_text(encoding="utf-8", errors="replace")[-6000:]
            logging.getLogger(__name__).error("Speaker worker failed (exit %s): %s", process.returncode, detail)
            error_path = job / "error.json"
            reason = read_json(error_path).get("message") if error_path.exists() else f"分析程序結束（代碼 {process.returncode}）"
            raise DiarizationError(f"說話者分析失敗：{reason}。原始逐字稿已保留。")
        return read_json(response)["segments"]
