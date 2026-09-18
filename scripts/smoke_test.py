"""Opt-in real inference test using a local spoken WAV (no audio network access).

Usage: python scripts/smoke_test.py --source .smoke/中文語音.wav --duration 3360
Download the tiny model first: python transcribe.py --download-model --model tiny
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from meeting_transcriber.audio import find_ffmpeg, probe_audio
from meeting_transcriber.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=80)
    parser.add_argument("--model", default="tiny")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scratch = root / ".smoke" / f"real-{args.duration}"
    scratch.mkdir(parents=True, exist_ok=True)
    config = load_config(root / "config.yaml")
    source = args.source.resolve()
    speech_duration = probe_audio(source).duration
    if args.duration < speech_duration * 2:
        parser.error("duration must fit the spoken sample twice")
    fixture = scratch / f"中文 長錄音 {args.duration}秒.m4a"
    pad_until = args.duration - speech_duration
    filters = f"[0:a]asplit=2[a][b];[a]apad=whole_dur={pad_until:.6f}[p];[p][b]concat=n=2:v=0:a=1[out]"
    subprocess.run([find_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source), "-filter_complex", filters, "-map", "[out]",
                    "-ar", "16000", "-c:a", "aac", str(fixture)], check=True)
    values = asdict(config)
    for key in ("input_dir", "output_dir", "logs_dir"):
        values[key] = str(scratch / key.removesuffix("_dir"))
    values["models_dir"] = str(root / "models")
    values.update(model=args.model, offline=True)
    config_path = scratch / "config.yaml"
    config_path.write_text(yaml.safe_dump(values, allow_unicode=True), encoding="utf-8")
    command = [sys.executable, str(root / "transcribe.py"), str(fixture), "--config", str(config_path)]
    started = time.monotonic()
    subprocess.run(command, check=True)
    elapsed = time.monotonic() - started
    folder = scratch / "output" / fixture.stem
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    transcript = json.loads((folder / f"{fixture.stem}.json").read_text(encoding="utf-8"))
    segments = transcript["segments"]
    assert metadata["completed"] is True
    assert abs(metadata["duration"] - args.duration) < 0.5
    assert len(metadata["generated"]) == 5
    assert segments and segments[0]["start"] < speech_duration
    assert segments[-1]["end"] > args.duration - speech_duration
    assert all(0 <= s["start"] < s["end"] <= metadata["duration"] for s in segments)
    assert all(a["end"] <= b["start"] for a, b in zip(segments, segments[1:]))
    for name in metadata["generated"]:
        assert (folder / name).read_text(encoding="utf-8").strip()
    repeated = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    assert "[SKIP]" in repeated.stdout
    summary = {"audio": str(fixture), "output": str(folder), "duration": metadata["duration"],
               "last_segment_end": segments[-1]["end"], "segments": len(segments),
               "processing_seconds": round(elapsed, 2), "device": metadata["device"],
               "model": metadata["model"], "all_outputs": True, "skip_verified": True,
               "description": "Local synthetic Chinese speech at start and end; silence in between."}
    (scratch / "validation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

