import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import time

from PySide6.QtCore import QCoreApplication, QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication
import pytest

from meeting_transcriber.gui import MainWindow
from meeting_transcriber.progress import ProgressEvent
from meeting_transcriber.transcriber import TranscriptionResult


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_drop_files_settings_and_progress(app, tmp_path, config):
    path = tmp_path / "中文 測試.m4a"
    path.write_bytes(b"test")
    window = MainWindow(config)
    window.show()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)), QUrl("https://example.com/private.wav")])
    enter = QDragEnterEvent(window.drop_zone.rect().center(), Qt.DropAction.CopyAction, mime,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QCoreApplication.sendEvent(window.drop_zone, enter)
    drop = QDropEvent(QPointF(50, 50), Qt.DropAction.CopyAction, mime,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QCoreApplication.sendEvent(window.drop_zone, drop)
    assert window.files == [path] and window.start_button.isEnabled()
    window.add_files([path])
    assert len(window.files) == 1
    assert not window.speakers.isEnabled()
    window.diarize.setChecked(True)
    assert window.speakers.isEnabled()
    window._file_started(0, path.name)
    window._progress(ProgressEvent("transcribing", 30, 60, "cpu"))
    assert window.bar.value() == 500 and "1/2" in window.stage.text()
    assert "00:30 / 01:00" in window.timeline.text()
    window._progress(ProgressEvent("diarizing", 1, 4, "cpu"))
    assert "2/2" in window.stage.text()
    window.close()


def test_worker_wires_selected_options_and_finishes(app, tmp_path, config, monkeypatch):
    observed = []
    class FakeEngine:
        def __init__(self, settings, warn, **kwargs):
            observed.append(settings)
            self.events = kwargs["events"]
            self.on_segment = kwargs["on_segment"]
        def transcribe(self, path, force):
            folder = config.output_dir / path.stem
            folder.mkdir(parents=True)
            (folder / f"{path.stem}.txt").write_text("【說話者 1】測試完成。", encoding="utf-8")
            self.events(ProgressEvent("transcribing", 1, 2, "cpu"))
            self.on_segment({"text": "測試完成。"})
            return TranscriptionResult(path, folder, 2, 1, 1, [path.stem + ".txt"])
    monkeypatch.setattr("meeting_transcriber.gui.Transcriber", FakeEngine)
    source = tmp_path / "中文.wav"
    source.write_bytes(b"test")
    window = MainWindow(config)
    window.add_files([source])
    window.diarize.setChecked(True)
    window.speakers.setValue(2)
    window.start()
    deadline = time.monotonic() + 5
    while window.worker and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert window.worker is None
    assert observed[0].diarize and observed[0].num_speakers == 2
    assert "測試完成" in window.preview.toPlainText()
    assert window.bar.value() == 1000 and window.start_button.isEnabled()
    window.close()


def test_switching_preview_does_not_mix_recordings(app, tmp_path, config):
    window = MainWindow(config)
    files = [tmp_path / "一.wav", tmp_path / "二.wav"]
    for path in files:
        path.write_bytes(b"audio")
    window.add_files(files)
    window._file_started(0, files[0].name)
    window._segment("第一個檔案的內容")
    window._file_started(1, files[1].name)
    window._segment("第二個檔案的內容")
    window.table.selectRow(0)
    assert window.preview.toPlainText() == "第一個檔案的內容"
    window.table.selectRow(1)
    assert window.preview.toPlainText() == "第二個檔案的內容"
    window.close()


def test_removing_stopped_file_keeps_preview_with_its_recording(app, tmp_path, config):
    window = MainWindow(config)
    files = [tmp_path / "一.wav", tmp_path / "二.wav"]
    for path in files:
        path.write_bytes(b"audio")
    window.add_files(files)
    for row, text in enumerate(("第一份未完成內容", "第二份未完成內容")):
        window._file_started(row, files[row].name)
        window._segment(text)
    window.table.selectRow(0)
    window.remove_selected()
    window.table.selectRow(0)
    assert window.files == [files[1]]
    assert window.preview.toPlainText() == "第二份未完成內容"
    window.remove_selected()
    assert window.preview.toPlainText() == ""
    assert not window.copy_button.isEnabled()
    window.close()


def test_missing_output_never_displays_previous_recording(app, tmp_path, config):
    source = tmp_path / "中文.wav"
    source.write_bytes(b"audio")
    window = MainWindow(config)
    window.add_files([source])
    window.preview.setPlainText("另一份逐字稿")
    window.results[0] = TranscriptionResult(source, tmp_path / "missing", 2, 1, 1, [])
    window.show_selected_result()
    assert window.preview.toPlainText() == ""
    assert not window.copy_button.isEnabled()
    assert "無法讀取預覽" in window.log.toPlainText()
    assert window.tabs.currentIndex() == 1
    window.close()
