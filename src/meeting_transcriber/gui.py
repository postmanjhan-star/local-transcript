"""Native Qt desktop UI. Recognition runs in a worker; all widgets stay on the GUI thread."""
from dataclasses import replace
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent, QFont, QFontDatabase
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)

from .audio import SUPPORTED_EXTENSIONS
from .config import Config, load_config, hotword_terms
from .errors import CancelledError
from .formatter import elapsed_label, expected_outputs
from .progress import ProgressEvent, TimeEstimator
from .transcriber import Transcriber, TranscriptionResult
from .utils import configure_logging, output_lock

PHASES = {"preparing": "準備檔案", "loading": "載入語音模型", "transcribing": "辨識語音",
          "speaker_download": "下載說話者模型", "diarizing": "分析不同說話者",
          "clustering": "整理說話者標記", "writing": "儲存逐字稿"}

STYLE = """
QMainWindow { background: #f3f6fa; }
QWidget { color: #20334b; font-size: 14px; font-family: 'Microsoft JhengHei UI', 'Segoe UI'; }
QLabel#eyebrow { color: #17857d; font-size: 11px; font-weight: 700; letter-spacing: 2px; }
QLabel#title { font-size: 28px; font-weight: 700; color: #142b44; }
QLabel#muted { color: #708198; font-size: 12px; }
QLabel#privacy { color: #08786d; background: #e0f3ee; border-radius: 13px; padding: 7px 13px; font-size: 12px; }
QFrame#card { background: white; border: 1px solid #dce4ee; border-radius: 12px; }
QFrame#drop { background: #eaf6f4; border: 2px dashed #86c5bb; border-radius: 12px; }
QFrame#drop QLabel { background: transparent; color: #216f68; }
QLabel#section { font-weight: 700; font-size: 14px; }
QPushButton { padding: 8px 14px; border: 1px solid #d1dce6; border-radius: 7px; background: white; }
QPushButton:hover { background: #eaf2f7; border-color: #a2bacf; }
QPushButton:disabled { color: #9ba9b7; background: #eff3f6; border-color: #e0e6ec; }
QPushButton#primary { color: white; background: #087f77; border-color: #087f77; font-weight: 700; }
QPushButton#primary:hover { background: #066b65; }
QPushButton#primary:disabled { background: #9ac9c2; border-color: #9ac9c2; }
QComboBox, QSpinBox { background: white; border: 1px solid #cedbe5; border-radius: 6px; padding: 5px 8px; min-height: 22px; }
QCheckBox { spacing: 7px; }
QTableWidget { background: white; border: none; gridline-color: #eef2f7; selection-background-color: #e6f3f1; selection-color: #20334b; }
QHeaderView::section { background: #f7f9fc; color: #6b7f94; padding: 7px; border: none; font-size: 12px; }
QTableWidget::item { padding: 6px; }
QPlainTextEdit { background: white; border: none; padding: 9px; selection-background-color: #c9e8e4; }
QTabWidget::pane { border: none; background: white; }
QTabBar::tab { background: #f0f4f8; color: #728399; padding: 8px 15px; }
QTabBar::tab:selected { color: #087f77; background: white; font-weight: 700; }
QProgressBar { background: #e7edf3; border: none; border-radius: 5px; min-height: 10px; max-height: 10px; }
QProgressBar::chunk { background: #15988e; border-radius: 5px; }
QSplitter::handle { background: transparent; width: 12px; }
"""


def label(text: str, name: str = "") -> QLabel:
    item = QLabel(text)
    item.setObjectName(name)
    return item


class DropZone(QFrame):
    files_dropped = Signal(list)
    browse_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(22, 16, 22, 16)
        icon = label("↓")
        icon.setStyleSheet("font-size: 36px; padding-right: 12px;")
        layout.addWidget(icon)
        copy = QVBoxLayout()
        copy.addWidget(label("把音檔拖到這裡", "section"))
        copy.addWidget(label("支援 M4A、MP3、WAV、MP4 等格式，可一次加入多個檔案", "muted"))
        layout.addLayout(copy, 1)
        browse = QPushButton("選擇檔案")
        browse.clicked.connect(self.browse_requested)
        layout.addWidget(browse)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        self.files_dropped.emit([Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()


class TranscriptionWorker(QThread):
    file_started = Signal(int, str)
    progress_changed = Signal(object)
    segment_ready = Signal(str)
    message = Signal(str)
    file_completed = Signal(int, object)
    file_failed = Signal(int, str)
    file_cancelled = Signal(int)

    def __init__(self, files: list[Path], config: Config, force: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.files, self.config, self.force = files, config, force
        self.stop_event = threading.Event()

    def run(self) -> None:
        active = 0
        try:
            with output_lock(self.config.output_dir):
                engine = Transcriber(self.config, self.message.emit,
                    events=self.progress_changed.emit, cancelled=self.stop_event.is_set,
                    on_segment=lambda s: self.segment_ready.emit(s["text"]))
                for index, path in enumerate(self.files):
                    active = index
                    if self.stop_event.is_set():
                        break
                    self.file_started.emit(index, path.name)
                    try:
                        result = engine.transcribe(path, self.force)
                        self.file_completed.emit(index, result)
                    except CancelledError:
                        self.file_cancelled.emit(index)
                        break
                    except Exception as exc:
                        logging.getLogger(__name__).exception("GUI transcription failed: %s", path)
                        self.file_failed.emit(index, str(exc))
        except Exception as exc:
            logging.getLogger(__name__).exception("GUI worker failed")
            self.file_failed.emit(active, str(exc))


class MainWindow(QMainWindow):
    def __init__(self, config: Config):
        super().__init__()
        # The Windows offscreen Qt backend does not enumerate system fonts.
        if not QFontDatabase.families():
            import os
            fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            for name in ("msjh.ttc", "msjhbd.ttc", "segoeui.ttf", "segoeuib.ttf"):
                if (fonts / name).is_file():
                    QFontDatabase.addApplicationFont(str(fonts / name))
        self.config = config
        self.files: list[Path] = []
        self.results: dict[int, TranscriptionResult] = {}
        self.live_text: dict[int, list[str]] = {}
        self.worker: TranscriptionWorker | None = None
        self.active_row = -1
        self.file_started_at = 0.0
        self.estimator = TimeEstimator()
        self.close_when_finished = False
        self.setWindowTitle("本機逐字稿 · Local Transcript")
        self.resize(1120, 850)
        self.setMinimumSize(920, 740)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE)
        self._build()
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

    def _build(self) -> None:
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(14)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(4)
        titles.addWidget(label("LOCAL TRANSCRIPT", "eyebrow"))
        titles.addWidget(label("把聲音，整理成文字。", "title"))
        titles.addWidget(label("拖入錄音，產生可搜尋、可複製的繁體中文逐字稿。", "muted"))
        header.addLayout(titles, 1)
        header.addWidget(label("●  全程本機處理", "privacy"), 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self.add_files)
        self.drop_zone.browse_requested.connect(self.browse_files)
        layout.addWidget(self.drop_zone)

        settings = QFrame()
        settings.setObjectName("card")
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(16, 12, 16, 12)
        choices = QHBoxLayout()
        choices.addWidget(label("辨識模型"))
        self.model = QComboBox()
        for name, description in [("tiny", "tiny · 快速測試"), ("small", "small · 較快"),
                                  ("medium", "medium · 平衡品質"), ("large-v3", "large-v3 · 較高品質")]:
            self.model.addItem(description, name)
        if self.model.findData(self.config.model) < 0:
            self.model.addItem(self.config.model, self.config.model)
        self.model.setCurrentIndex(self.model.findData(self.config.model))
        choices.addWidget(self.model)
        choices.addSpacing(16)
        self.diarize = QCheckBox("區分不同人說話")
        self.diarize.setChecked(self.config.diarize)
        choices.addWidget(self.diarize)
        choices.addWidget(label("人數"))
        self.speakers = QSpinBox()
        self.speakers.setRange(0, 20)
        self.speakers.setSpecialValueText("自動")
        self.speakers.setValue(self.config.num_speakers)
        self.speakers.setEnabled(self.config.diarize)
        self.speakers.setMinimumWidth(72)
        self.diarize.toggled.connect(self.speakers.setEnabled)
        choices.addWidget(self.speakers)
        choices.addStretch()
        self.offline = QCheckBox("離線模式")
        self.offline.setChecked(self.config.offline)
        self.offline.setToolTip("只用已下載的模型；未快取時會提示錯誤。")
        choices.addWidget(self.offline)
        self.force = QCheckBox("重新轉錄")
        self.force.setToolTip("先備份既有結果，再從頭辨識。一般重試或切換說話者選項不需要勾選。")
        choices.addWidget(self.force)
        settings_layout.addLayout(choices)
        terms_row = QHBoxLayout()
        terms_row.addWidget(label("辨識關鍵詞"))
        self.hotwords = QLineEdit()
        self.hotwords.setText(self.config.hotwords)
        self.hotwords.setPlaceholderText("選填：補充這次會議的專有名詞，不必重複預設詞庫")
        self.hotwords.setToolTip("自動載入 config.yaml 的 vocabulary、terminology 正確寫法，以及本機 vocabulary.local.txt；這裡只需額外補充。")
        terms_row.addWidget(self.hotwords, 1)
        self.vocabulary_label = label(f"自動詞庫 {len(hotword_terms(self.config))} 詞")
        self.vocabulary_label.setToolTip("包含通用製造詞彙及本地私有詞彙；輸入框只用來額外補充。")
        terms_row.addWidget(self.vocabulary_label)
        self.context = QCheckBox("連貫上下文（實驗性）")
        self.context.setChecked(self.config.condition_on_previous_text)
        self.context.setToolTip("可能改善連貫性，也可能導致重複或幻覺；建議先維持關閉。")
        terms_row.addWidget(self.context)
        settings_layout.addLayout(terms_row)
        settings_layout.addWidget(label("說話者以 1、2… 標示，不辨識姓名；短句或同時發言可能不準確。首次使用會下載模型。", "muted"))
        layout.addWidget(settings)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        queue_card = QFrame()
        queue_card.setObjectName("card")
        queue_layout = QVBoxLayout(queue_card)
        queue_layout.setContentsMargins(12, 10, 12, 10)
        queue_header = QHBoxLayout()
        self.queue_title = label("檔案清單  ·  0", "section")
        queue_header.addWidget(self.queue_title, 1)
        self.remove_button = QPushButton("移除")
        self.remove_button.clicked.connect(self.remove_selected)
        queue_header.addWidget(self.remove_button)
        queue_layout.addLayout(queue_header)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["錄音檔案", "狀態"])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.show_selected_result)
        queue_layout.addWidget(self.table)
        queue_layout.addWidget(label("依序處理檔案；已完成的錄音會自動跳過。", "muted"))
        splitter.addWidget(queue_card)
        preview_card = QFrame()
        preview_card.setObjectName("card")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(12, 10, 12, 10)
        preview_header = QHBoxLayout()
        preview_header.addWidget(label("逐字稿預覽", "section"), 1)
        self.copy_button = QPushButton("複製文字")
        self.copy_button.clicked.connect(lambda: QApplication.clipboard().setText(self.preview.toPlainText()))
        self.copy_button.setEnabled(False)
        preview_header.addWidget(self.copy_button)
        preview_layout.addLayout(preview_header)
        self.tabs = QTabWidget()
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setAcceptDrops(False)
        self.preview.setPlaceholderText("轉錄開始後，辨識出的文字會陸續出現在這裡。\n\n完成後可複製全文，或開啟輸出資料夾。")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setAcceptDrops(False)
        self.tabs.addTab(self.preview, "文字內容")
        self.tabs.addTab(self.log, "處理訊息")
        preview_layout.addWidget(self.tabs)
        splitter.addWidget(preview_card)
        splitter.setSizes([430, 600])
        layout.addWidget(splitter, 1)

        progress_card = QFrame()
        progress_card.setObjectName("card")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(16, 12, 16, 12)
        current = QHBoxLayout()
        self.stage = label("準備就緒", "section")
        current.addWidget(self.stage, 1)
        self.percent = label("", "muted")
        current.addWidget(self.percent)
        progress_layout.addLayout(current)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        progress_layout.addWidget(self.bar)
        metrics = QHBoxLayout()
        self.timeline = label("加入檔案後即可開始", "muted")
        self.elapsed = label("此檔耗時  —", "muted")
        self.eta = label("本階段預估剩餘  —", "muted")
        metrics.addWidget(self.timeline, 1)
        metrics.addWidget(self.elapsed)
        metrics.addSpacing(24)
        metrics.addWidget(self.eta)
        progress_layout.addLayout(metrics)
        layout.addWidget(progress_card)
        actions = QHBoxLayout()
        self.open_button = QPushButton("開啟輸出資料夾")
        self.open_button.clicked.connect(self.open_output)
        actions.addWidget(self.open_button)
        actions.addStretch()
        self.stop_button = QPushButton("停止並保留進度")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        actions.addWidget(self.stop_button)
        self.start_button = QPushButton("開始轉錄")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumWidth(136)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start)
        actions.addWidget(self.start_button)
        layout.addLayout(actions)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self.worker is None and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        self.add_files([Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    @Slot()
    def browse_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "選擇錄音", str(self.config.input_dir),
                                               "音訊與影片 (" + " ".join("*" + e for e in sorted(SUPPORTED_EXTENSIONS)) + ")")
        self.add_files([Path(p) for p in paths])

    @Slot(list)
    def add_files(self, paths: list[Path]) -> None:
        if self.worker:
            return
        rejected = []
        for value in paths:
            path = Path(value).resolve()
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                rejected.append(path.name)
                continue
            if path in self.files:
                continue
            row = len(self.files)
            self.files.append(path)
            self.table.insertRow(row)
            name = QTableWidgetItem(path.name)
            name.setToolTip(str(path))
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, QTableWidgetItem("等待中"))
            self.table.setRowHeight(row, 42)
        self.queue_title.setText(f"檔案清單  ·  {len(self.files)}")
        self.start_button.setEnabled(bool(self.files))
        if self.files and self.table.currentRow() < 0:
            self.table.selectRow(0)
        if rejected:
            self.log.appendPlainText("無法加入（請使用支援格式的本機檔案）：" + "、".join(rejected))
            self.stage.setText("部分檔案未加入，請查看處理訊息")
            self.tabs.setCurrentIndex(1)

    @Slot()
    def remove_selected(self) -> None:
        row = self.table.currentRow()
        if self.worker or row < 0:
            return
        self.files.pop(row)
        self.results = {(i - 1 if i > row else i): result for i, result in self.results.items() if i != row}
        self.live_text = {(i - 1 if i > row else i): text for i, text in self.live_text.items() if i != row}
        self.active_row = -1
        self.table.removeRow(row)
        self.queue_title.setText(f"檔案清單  ·  {len(self.files)}")
        self.start_button.setEnabled(bool(self.files))
        self.show_selected_result()

    def _set_running(self, running: bool) -> None:
        for control in (self.drop_zone, self.model, self.diarize, self.offline, self.force,
                        self.hotwords, self.context, self.remove_button):
            control.setEnabled(not running)
        self.speakers.setEnabled(not running and self.diarize.isChecked())
        self.start_button.setEnabled(not running and bool(self.files))
        self.stop_button.setEnabled(running)
        self.stop_button.setText("停止並保留進度")

    @Slot()
    def start(self) -> None:
        if self.worker or not self.files:
            return
        settings = replace(self.config, model=self.model.currentData(), diarize=self.diarize.isChecked(),
                           num_speakers=self.speakers.value(), offline=self.offline.isChecked(),
                            hotwords=self.hotwords.text().strip(),
                            condition_on_previous_text=self.context.isChecked())
        self.log.clear()
        self.results.clear()
        self.live_text.clear()
        for i in range(len(self.files)):
            self.table.item(i, 1).setText("等待中")
        self._set_running(True)
        worker = TranscriptionWorker(list(self.files), settings, self.force.isChecked(), self)
        self.worker = worker
        worker.file_started.connect(self._file_started)
        worker.progress_changed.connect(self._progress)
        worker.segment_ready.connect(self._segment)
        worker.message.connect(self.log.appendPlainText)
        worker.file_completed.connect(self._file_completed)
        worker.file_failed.connect(self._file_failed)
        worker.file_cancelled.connect(self._file_cancelled)
        worker.finished.connect(self._finished)
        worker.finished.connect(worker.deleteLater)
        self.timer.start()
        worker.start()

    @Slot(int, str)
    def _file_started(self, row: int, name: str) -> None:
        self.active_row = row
        self.live_text[row] = []
        self.file_started_at = time.monotonic()
        self.estimator = TimeEstimator()
        self.table.selectRow(row)
        self.preview.clear()
        self.copy_button.setEnabled(False)
        self.table.item(row, 1).setText("處理中")
        self.log.appendPlainText(f"\n[{row + 1}/{len(self.files)}] {name}")

    @Slot(object)
    def _progress(self, event: ProgressEvent) -> None:
        self.estimator.update(event)
        phase = PHASES.get(event.phase, event.phase)
        if self.diarize.isChecked() and event.phase in {"transcribing", "diarizing"}:
            phase = ("1/2 · " if event.phase == "transcribing" else "2/2 · ") + phase
        suffix = f" · {event.device.upper()}" if event.device else ""
        self.stage.setText(f"檔案 {self.active_row + 1}/{len(self.files)}　{phase}{suffix}")
        measurable = event.total > 0 and event.phase in {"transcribing", "diarizing", "speaker_download"}
        if measurable:
            self.bar.setRange(0, 1000)
            self.bar.setValue(round(min(1, event.done / event.total) * 1000))
            self.percent.setText(f"{event.done / event.total:.0%}")
            if event.phase == "transcribing":
                self.timeline.setText(f"音訊 {elapsed_label(event.done)} / {elapsed_label(event.total)}")
            elif event.phase == "speaker_download":
                self.timeline.setText(f"模型 {event.done / 1e6:.1f} / {event.total / 1e6:.1f} MB")
            else:
                self.timeline.setText(f"音訊 {elapsed_label(event.done)} / {elapsed_label(event.total)}")
        else:
            self.bar.setRange(0, 0)
            self.percent.setText("")
            self.timeline.setText("首次載入可能較久，已完成的轉錄會保留。" if event.phase == "loading" else phase)
        self._tick()

    @Slot()
    def _tick(self) -> None:
        if self.file_started_at:
            self.elapsed.setText(f"此檔耗時  {elapsed_label(time.monotonic() - self.file_started_at)}")
        remaining = self.estimator.remaining()
        self.eta.setText("本階段預估剩餘  " + (f"約 {elapsed_label(remaining)}" if remaining is not None else "估算中…"))

    @Slot(str)
    def _segment(self, text: str) -> None:
        self.live_text.setdefault(self.active_row, []).append(text)
        if self.table.currentRow() == self.active_row:
            self.preview.appendPlainText(text + "\n")
            self.copy_button.setEnabled(True)

    @Slot(int, object)
    def _file_completed(self, row: int, result: TranscriptionResult) -> None:
        self.results[row] = result
        self.table.item(row, 1).setText("已完成（略過）" if result.skipped else "已完成")
        self.log.appendPlainText(f"完成：{result.folder}")
        if self.table.currentRow() == row:
            self.show_selected_result()

    @Slot(int, str)
    def _file_failed(self, row: int, message: str) -> None:
        self.table.item(row, 1).setText("失敗")
        self.log.appendPlainText(f"錯誤：{message}\n已完成的片段保留，可再次執行。詳細記錄：{self.config.logs_dir}")
        self.tabs.setCurrentIndex(1)

    @Slot(int)
    def _file_cancelled(self, row: int) -> None:
        self.table.item(row, 1).setText("已停止，可續跑")
        self.log.appendPlainText("已停止，已完成的片段與分析進度保留。")

    @Slot()
    def _finished(self) -> None:
        stopped = self.worker.stop_event.is_set() if self.worker else False
        self.worker = None
        self.timer.stop()
        self._set_running(False)
        self.bar.setRange(0, 1000)
        if len(self.results) == len(self.files):
            self.bar.setValue(1000)
            self.percent.setText("100%")
            self.stage.setText("全部完成，可以開啟逐字稿了")
        else:
            self.stage.setText("已停止，再次開始即可續跑" if stopped else "部分檔案失敗，請查看處理訊息")
            self.percent.setText("")
        self.eta.setText("本階段預估剩餘  —")
        self.timeline.setText(f"已完成 {len(self.results)} / {len(self.files)} 個檔案")
        if self.close_when_finished:
            self.close()

    @Slot()
    def stop(self) -> None:
        if self.worker:
            self.worker.stop_event.set()
            self.stop_button.setEnabled(False)
            self.stop_button.setText("正在停止…")
            self.log.appendPlainText("停止要求已送出，會在目前下載／解碼／辨識步驟返回後保留進度。")

    @Slot()
    def show_selected_result(self) -> None:
        row = self.table.currentRow()
        result = self.results.get(row)
        if result:
            paths = expected_outputs(result.folder, result.source.stem, self.config.output)
            target = paths.get("txt") or paths.get("timestamp_txt") or paths.get("markdown")
            try:
                self.preview.setPlainText(target.read_text(encoding="utf-8") if target else "逐字稿已輸出至資料夾。")
            except OSError as exc:
                self.preview.clear()
                self.log.appendPlainText(f"無法讀取預覽：{exc}")
                self.tabs.setCurrentIndex(1)
            self.copy_button.setEnabled(bool(self.preview.toPlainText()))
        elif row in self.live_text:
            self.preview.setPlainText("\n\n".join(self.live_text[row]))
            self.copy_button.setEnabled(bool(self.preview.toPlainText()))
        else:
            self.preview.clear()
            self.copy_button.setEnabled(False)

    @Slot()
    def open_output(self) -> None:
        result = self.results.get(self.table.currentRow())
        folder = result.folder if result else self.config.output_dir
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker:
            answer = QMessageBox.question(self, "轉錄仍在進行", "停止並保留進度後關閉？目前辨識步驟結束後才會關閉視窗。")
            if answer == QMessageBox.StandardButton.Yes:
                self.close_when_finished = True
                self.stop()
            event.ignore()
        else:
            event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Local Transcript")
    app.setFont(QFont("Microsoft JhengHei UI", 10))
    try:
        config = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
        for folder in (config.input_dir, config.output_dir, config.models_dir, config.logs_dir):
            folder.mkdir(parents=True, exist_ok=True)
        configure_logging(config.logs_dir)
        window = MainWindow(config)
        if len(sys.argv) > 1:
            window.add_files([Path(p) for p in sys.argv[1:]])
        window.show()
        return app.exec()
    except Exception as exc:
        logging.getLogger(__name__).exception("GUI startup failed")
        QMessageBox.critical(None, "無法啟動", f"{exc}\n\n請重新執行 setup.ps1。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
