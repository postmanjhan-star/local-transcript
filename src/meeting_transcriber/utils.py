"""Durable writes, input identity, output selection, and process locking."""
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator
import uuid

from .errors import CheckpointError, TranscriberError


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, data: Any) -> None:
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected object")
        return value
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"無法讀取 {path}，原檔已保留。可用 --force 備份後重跑。") from exc


def source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"source_path": str(path.resolve()), "source_file": path.name,
            "size": stat.st_size, "mtime": stat.st_mtime, "mtime_ns": stat.st_mtime_ns}


def same_source(metadata: dict[str, Any], identity: dict[str, Any]) -> bool:
    return all(metadata.get(key) == identity.get(key)
               for key in ("source_path", "source_file", "size", "mtime_ns"))


def already_processed(metadata: dict[str, Any], identity: dict[str, Any], fingerprint: str) -> bool:
    return metadata.get("completed") is True and same_source(metadata, identity) and metadata.get("config_hash") == fingerprint


def output_path(source: Path, root: Path) -> Path:
    """Keep the usual human-readable stem; disambiguate equal stems safely."""
    root = root.resolve()
    stem = source.stem.rstrip(" .") or "recording"
    simple = root / stem
    if not simple.exists():
        return simple
    metadata = simple / "metadata.json"
    if metadata.exists():
        try:
            if read_json(metadata).get("source_path") == str(source.resolve()):
                return simple
        except CheckpointError:
            pass
    digest = hashlib.sha256(str(source.resolve()).casefold().encode("utf-8")).hexdigest()[:12]
    candidate = root / f"{stem}--{digest}"
    if candidate.exists() and (candidate / "metadata.json").exists():
        if read_json(candidate / "metadata.json").get("source_path") != str(source.resolve()):
            raise CheckpointError(f"輸出目錄衝突：{candidate}")
    return candidate


def archive_results(folder: Path) -> Path | None:
    """Never destroy existing results, including when --force is requested."""
    root = folder.resolve()
    entries = [p for p in root.iterdir() if p.name != "history"] if root.exists() else []
    if not entries:
        return None
    target = root / "history" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    if not target.resolve().is_relative_to(root):
        raise CheckpointError("備份目錄超出輸出路徑。")
    for entry in entries:
        if entry.is_symlink() or not entry.resolve().is_relative_to(root):
            raise CheckpointError(f"拒絕搬移輸出目錄外的連結：{entry}")
    target.mkdir(parents=True)
    for entry in entries:
        shutil.move(str(entry), str(target / entry.name))
    return target


def load_partial(path: Path) -> list[dict[str, Any]]:
    """Recover a torn final write; refuse to silently discard interior corruption."""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    segments: list[dict[str, Any]] = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        try:
            segment = json.loads(line)
            import math
            if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
                raise ValueError("Invalid segment")
            start, end = float(segment["start"]), float(segment["end"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError("Invalid timestamps")
            if segments and start < float(segments[-1]["end"]) - 0.001:
                raise ValueError("Non-monotonic timestamps")
        except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise CheckpointError(f"checkpoint 第 {index + 1} 行損毀，已保留原檔：{path}") from exc
            backup = path.with_name(path.name + ".damaged-" + uuid.uuid4().hex[:8])
            shutil.copy2(path, backup)
            atomic_write(path, data[:valid_bytes].decode("utf-8"))
            logging.getLogger(__name__).warning("Recovered torn final checkpoint line; backup=%s", backup)
            break
        segments.append(segment)
        valid_bytes += len(line)
    if segments and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        # A complete JSON object without its newline is still a valid committed segment.
        atomic_write(path, path.read_text(encoding="utf-8") + "\n")
    return segments


def append_segment(path: Path, segment: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(segment, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def output_lock(root: Path) -> Iterator[None]:
    """OS-held lock releases automatically on process death (no stale PID files)."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".transcribe.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise TranscriberError("另一個轉錄程序正在使用此輸出目錄，請等它完成。") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def configure_logging(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"transcribe-{datetime.now():%Y%m%d}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    for previous in root.handlers[:]:
        root.removeHandler(previous)
        previous.close()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return path

