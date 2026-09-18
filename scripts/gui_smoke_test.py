"""Run the actual Qt worker with local two-voice fixtures and save UI evidence.

Requires .smoke/speakers/female.wav and male.wav (local speech synthesis),
plus cached tiny and speaker models. Does not use the network.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONUTF8", "1")

from collections import Counter
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication

from meeting_transcriber.audio import find_ffmpeg, probe_audio
from meeting_transcriber.config import load_config
from meeting_transcriber.gui import MainWindow
from meeting_transcriber.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scratch = root / ".smoke" / "speakers"
    female, male = scratch / "female.wav", scratch / "male.wav"
    destination = scratch / "兩人 中文會議.m4a"
    subprocess.run([find_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(female), "-i", str(male), "-i", str(female), "-i", str(male),
                    "-filter_complex", "[0:a][1:a][2:a][3:a]concat=n=4:v=0:a=1[out]",
                    "-map", "[out]", "-ar", "16000", "-ac", "1", "-c:a", "aac", str(destination)], check=True)
    config = replace(load_config(root / "config.yaml"), model=args.model, diarize=True, num_speakers=2,
                     offline=True, output_dir=scratch / "output", logs_dir=scratch / "logs")
    config.output_dir.mkdir(exist_ok=True)
    configure_logging(config.logs_dir)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(config)
    window.add_files([destination])
    window.show()
    app.processEvents()
    window.grab().save(str(scratch / "ui-ready.png"))
    window.start()
    observed = []
    window.worker.progress_changed.connect(lambda event: observed.append(event.phase))
    deadline = time.monotonic() + 600
    while window.worker and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.03)
    if window.worker:
        window.stop()
        while window.worker:
            app.processEvents()
            time.sleep(.05)
        raise AssertionError("UI worker timed out")
    assert len(window.results) == 1, window.log.toPlainText()
    result = window.results[0]
    text = window.preview.toPlainText()
    assert "【說話者 1】" in text and "【說話者 2】" in text, text
    assert window.bar.value() == 1000 and window.start_button.isEnabled()
    assert "transcribing" in observed and "diarizing" in observed
    data = json.loads((result.folder / "diarization.json").read_text(encoding="utf-8"))
    length_a, length_b = probe_audio(female).duration, probe_audio(male).duration
    bounds = [0, length_a, length_a + length_b, 2 * length_a + length_b, 2 * (length_a + length_b)]
    dominant = []
    for left, right in zip(bounds, bounds[1:]):
        votes = [row["speaker"] for row in data["windows"] if left + 1 < (row["start"] + row["end"]) / 2 < right - 1]
        assert votes, (left, right)
        dominant.append(Counter(votes).most_common(1)[0][0])
    assert dominant[0] == dominant[2] and dominant[1] == dominant[3] and dominant[0] != dominant[1], dominant
    assert len(result.generated) == 5
    app.processEvents()
    window.grab().save(str(scratch / "ui-completed.png"))
    summary = {"model": args.model, "duration": result.duration, "processing_seconds": result.processing_seconds,
               "speakers": data["speakers"], "dominant_speakers_by_turn": dominant,
               "all_five_outputs": True, "gui_worker_completed": True,
               "observed_phases": sorted(set(observed)), "output": str(result.folder)}
    (scratch / "gui-validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    window.close()


if __name__ == "__main__":
    main()
