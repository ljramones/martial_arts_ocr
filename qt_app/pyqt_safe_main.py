# qt_app/pyqt_safe_main.py
import os

os.environ.setdefault("QT_PLUGIN_PATH", "")
os.environ.setdefault("QT_MEDIA_BACKEND", "none")
os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
os.environ.setdefault("QT_OPENGL", "software")

import sys
import logging
from pathlib import Path
from typing import List, Set

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal as Signal, QTimer, QMetaObject, Q_ARG
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QCheckBox, QMessageBox, QSplitter,
    QPlainTextEdit, QSizePolicy
)

from PIL import Image, ImageOps
from qt_app import orchestrator
from qt_app.orchestrator import Orchestrator, IMG_EXTS


# ---------------- Custom Log Handler for Qt ----------------
class QtLogHandler(logging.Handler):
    """Thread-safe log handler that sends logs to Qt widget."""

    def __init__(self, log_widget):
        super().__init__()
        self.log_widget = log_widget

    def emit(self, record):
        try:
            msg = self.format(record)
            # Use Qt's thread-safe method to add text
            QMetaObject.invokeMethod(
                self.log_widget,
                "appendPlainText",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, msg)
            )
        except Exception:
            self.handleError(record)


# ---------------- Workers ----------------
class AnalyzeWorker(QObject):
    step = Signal(str)
    finished = Signal(list)  # list[str]
    failed = Signal(str)

    def __init__(self, orchestrator: Orchestrator, image_path: str):
        super().__init__()
        self.orch = orchestrator
        self.image_path = image_path

    def run(self):
        try:
            self.step.emit("Starting analysis…")
            langs = self.orch.analyze_types(self.image_path)  # Set[str]
            res = sorted(langs)
            self.step.emit(f"Analyzer result: {res}")
            self.finished.emit(res)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class ProcessWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(str)  # doc_dir path
    failed = Signal(str)

    def __init__(self, orchestrator: Orchestrator, image_path: str, lang_keys: Set[str]):
        super().__init__()
        self.orch = orchestrator
        self.image_path = image_path
        self.lang_keys = lang_keys

    def run(self):
        try:
            def prog(pct: int, msg: str):
                self.progress.emit(int(pct), str(msg))

            res = self.orch.process(self.image_path, self.lang_keys, progress=prog)
            self.finished.emit(str(res.doc_dir))
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


# ---------------- Main Window ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Martial Arts OCR — PyQt6 UI")
        self.resize(1300, 850)
        self.orch = Orchestrator(processed_root="processed")

        splitter = QSplitter(self)
        left = QWidget();
        mid = QWidget();
        right = QWidget()
        left_l = QVBoxLayout(left);
        mid_l = QVBoxLayout(mid);
        right_l = QVBoxLayout(right)

        # Left: folder + files
        self.dir_label = QLabel("No folder chosen")
        self.btn_choose_dir = QPushButton("Choose Source Folder…")
        self.files_list = QListWidget()
        left_l.addWidget(self.dir_label);
        left_l.addWidget(self.btn_choose_dir);
        left_l.addWidget(self.files_list)

        # Middle: image preview (QLabel)
        self.preview = QLabel("Select an image…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setMinimumSize(400, 400)

        mid_buttons = QHBoxLayout()
        self.btn_recognize = QPushButton("Recognize Types")
        self.btn_process = QPushButton("Process")
        self.btn_process.setEnabled(False)
        mid_buttons.addWidget(self.btn_recognize)
        mid_buttons.addWidget(self.btn_process)

        mid_l.addWidget(self.preview, stretch=1)
        mid_l.addLayout(mid_buttons)

        # Right: checkboxes + sniff toggle + log
        self.chk_sniff = QCheckBox("Use OCR sniff (eng+jpn)")
        self.chk_sniff.setChecked(orchestrator.USE_TESSERACT_SNIFF)
        self.chk_sniff.stateChanged.connect(self._toggle_sniff)

        self.chk_en = QCheckBox("English")
        self.chk_jp_mod = QCheckBox("Japanese (Modern)")
        self.chk_jp_cls = QCheckBox("Japanese (Classical)")
        for cb in (self.chk_en, self.chk_jp_mod, self.chk_jp_cls):
            cb.stateChanged.connect(self._update_process_enabled)

        self.log_out = QPlainTextEdit()
        self.log_out.setReadOnly(True)
        self.log_out.setMaximumBlockCount(2000)
        self.status_lbl = QLabel("Ready.")

        right_l.addWidget(QLabel("Detected / Override:"))
        right_l.addWidget(self.chk_sniff)
        right_l.addWidget(self.chk_en)
        right_l.addWidget(self.chk_jp_mod)
        right_l.addWidget(self.chk_jp_cls)
        right_l.addSpacing(10)
        right_l.addWidget(QLabel("Log"))
        right_l.addWidget(self.log_out, stretch=1)
        right_l.addSpacing(10)
        right_l.addWidget(self.status_lbl)

        splitter.addWidget(left)
        splitter.addWidget(mid)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)

        # Set up logging to GUI
        self._setup_logging()

        # State
        self.current_dir: Path | None = None
        self.current_image: Path | None = None

        # Dedicated thread/worker refs to avoid GC and allow clean shutdown
        self.an_thread: QThread | None = None
        self.an_worker: AnalyzeWorker | None = None
        self.proc_thread: QThread | None = None
        self.proc_worker: ProcessWorker | None = None

        self._an_watchdog: QTimer | None = None

        # Signals
        self.btn_choose_dir.clicked.connect(self.choose_dir)
        self.files_list.itemSelectionChanged.connect(self.on_file_selected)
        self.btn_recognize.clicked.connect(self.on_recognize_clicked)
        self.btn_process.clicked.connect(self.on_process_clicked)

    def _setup_logging(self):
        """Configure logging to capture all module loggers and send to GUI."""
        # Get the root logger
        root_logger = logging.getLogger()

        # Remove any existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create console handler for debugging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Only warnings and above to console
        console_formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Create GUI handler
        gui_handler = QtLogHandler(self.log_out)
        gui_handler.setLevel(logging.INFO)  # INFO and above to GUI
        gui_formatter = logging.Formatter('[%(name)s] %(message)s')
        gui_handler.setFormatter(gui_formatter)
        root_logger.addHandler(gui_handler)

        # Set overall logging level
        root_logger.setLevel(logging.DEBUG)

        # Also set specific module loggers to ensure they're captured
        for module_name in ['processors.ocr_postprocessor', 'processors.ocr_processor',
                            'processors.ocr_engines', 'utils.image', 'utils.image.preprocessing',
                            'utils.image.layout', 'utils.image.regions', 'utils.image.io',
                            'utils.text_utils']:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.INFO)

    # ---------- lifecycle ----------
    def closeEvent(self, event):
        # Gracefully stop any running threads to prevent "QThread destroyed while running"
        for t in (self.an_thread, self.proc_thread):
            if t and t.isRunning():
                t.requestInterruption()
                t.quit()
                t.wait(3000)
        super().closeEvent(event)

    # ---------- logging / config ----------
    def log(self, msg: str):
        """Direct log method for UI events."""
        self.log_out.appendPlainText(msg)

    def _toggle_sniff(self):
        orchestrator.USE_TESSERACT_SNIFF = self.chk_sniff.isChecked()
        self.log(f"[config] USE_TESSERACT_SNIFF={orchestrator.USE_TESSERACT_SNIFF}")

    # ---------- directory + list ----------
    def choose_dir(self):
        # Start from current working directory instead of home
        start_dir = str(Path.cwd())  # Current working directory
        d = QFileDialog.getExistingDirectory(self, "Select source folder", start_dir)
        if not d:
            return
        self.current_dir = Path(d)
        self.dir_label.setText(str(self.current_dir))
        self.populate_files_list()

    def populate_files_list(self):
        self.files_list.clear()
        cnt = 0
        if self.current_dir:
            for p in sorted(self.current_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.files_list.addItem(str(p))
                    cnt += 1
        self.status_lbl.setText(f"Found {cnt} image(s).")
        self.log(f"[list] {cnt} images from {self.current_dir}")

    # ---------- selection + preview ----------
    def on_file_selected(self):
        for cb in (self.chk_en, self.chk_jp_mod, self.chk_jp_cls):
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        items = self.files_list.selectedItems()
        if not items:
            self.current_image = None
            self.preview.setText("Select an image…")
            self._update_process_enabled()
            return
        self.current_image = Path(items[0].text())
        self.log(f"[select] {self.current_image}")
        self.load_image_preview(self.current_image)
        self._update_process_enabled()

    def load_image_preview(self, img_path: Path):
        try:
            with Image.open(str(img_path)) as im:
                im = ImageOps.exif_transpose(im).convert("RGBA")
                qimg = QImage(im.tobytes("raw", "RGBA"), im.width, im.height, QImage.Format.Format_RGBA8888)
            pix = QPixmap.fromImage(qimg)
            self.preview.setPixmap(
                pix.scaled(self.preview.size(),
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            )
        except Exception as e:
            self.log(f"[preview] error: {e}")
            self.preview.setText(f"Preview error:\n{e}")

    # ---------- recognize (thread-safe, re-click-safe) ----------
    def on_recognize_clicked(self):
        if not self.current_image:
            QMessageBox.information(self, "No file", "Select an image.")
            return
        # Guard: don't start if already running
        if self.an_thread and self.an_thread.isRunning():
            self.log("[analyze] already running; ignoring extra click")
            return

        self.status_lbl.setText("Recognizing…")
        self.log("=== Recognize Types ===")
        self.log(f"File: {self.current_image}")
        self.btn_recognize.setEnabled(False)  # prevent double-starts

        # Build new thread/worker pair
        self.an_thread = QThread()
        self.an_worker = AnalyzeWorker(self.orch, str(self.current_image))
        self.an_worker.moveToThread(self.an_thread)

        # Connect
        self.an_thread.started.connect(self.an_worker.run)
        self.an_worker.step.connect(lambda m: self.log(f"[analyze] {m}"))
        self.an_worker.finished.connect(self._on_analyze_finished)
        self.an_worker.failed.connect(self._on_worker_failed)

        # Cleanup
        self.an_worker.finished.connect(self.an_thread.quit)
        self.an_worker.failed.connect(self.an_thread.quit)
        self.an_thread.finished.connect(self._an_analyze_cleanup)

        # Start
        self.an_thread.start()

        # Watchdog
        self._an_watchdog = QTimer(self)
        self._an_watchdog.setSingleShot(True)
        self._an_watchdog.timeout.connect(self._on_analyze_watchdog)
        self._an_watchdog.start(15000)

    def _on_analyze_watchdog(self):
        self.log("[watchdog] analyzer still running after 15s…")
        self.status_lbl.setText("Analyzer slow…")

    def _on_analyze_finished(self, lang_keys: List[str]):
        if self._an_watchdog:
            self._an_watchdog.stop()
        self.chk_en.setChecked("english" in lang_keys)
        self.chk_jp_mod.setChecked("jp_modern" in lang_keys)
        self.chk_jp_cls.setChecked("jp_classical" in lang_keys)
        self.status_lbl.setText(f"Recognized: {', '.join(lang_keys) if lang_keys else 'none'}")
        self._update_process_enabled()
        self.log(f"[analyze] done -> {lang_keys}")
        dbg = self.orch.last_analyze_debug()
        if dbg:
            self.log(
                f"[analyze] sniff_len={dbg.get('sniff_len')} "
                f"ratios: latin={dbg.get('latin_ratio')} jp={dbg.get('jp_ratio')}  "
                f"counts: latin={dbg.get('latin')} hira={dbg.get('hira')} "
                f"kata={dbg.get('kata')} kanji={dbg.get('kanji')}"
            )
            self.log(
                f"[analyze] threshold JP={dbg.get('JP_RATIO_THRESHOLD')} "
                f"sniff={dbg.get('USE_TESSERACT_SNIFF')}"
            )
        else:
            self.log("[analyze] no debug snapshot (sniff off or empty text)")

    def _an_analyze_cleanup(self):
        # thread finished; clear refs and re-enable button
        self.an_thread = None
        self.an_worker = None
        self.btn_recognize.setEnabled(True)

    # ---------- process (thread-safe) ----------
    def on_process_clicked(self):
        if not self.current_image:
            QMessageBox.information(self, "No file", "Select an image first.")
            return
        selected = self._selected_lang_keys()
        if not selected:
            QMessageBox.information(self, "Choose types", "Pick at least one language.")
            return

        # Guard: avoid overlapping process runs
        if self.proc_thread and self.proc_thread.isRunning():
            self.log("[process] already running; ignoring extra click")
            return

        self.status_lbl.setText(f"Processing as: {', '.join(sorted(selected))}…")
        self.log("=== Process ===")
        self.log(f"File: {self.current_image}")
        self.log(f"Langs: {sorted(selected)}")
        self.btn_process.setEnabled(False)

        self.proc_thread = QThread()
        self.proc_worker = ProcessWorker(self.orch, str(self.current_image), selected)
        self.proc_worker.moveToThread(self.proc_thread)

        self.proc_thread.started.connect(self.proc_worker.run)
        self.proc_worker.progress.connect(lambda pct, msg: self.log(f"[process] {pct}% {msg}"))
        self.proc_worker.finished.connect(self._on_process_finished)
        self.proc_worker.failed.connect(self._on_worker_failed)

        self.proc_worker.finished.connect(self.proc_thread.quit)
        self.proc_worker.failed.connect(self.proc_thread.quit)
        self.proc_thread.finished.connect(self._an_process_cleanup)

        self.proc_thread.start()

    def _on_process_finished(self, doc_dir: str):
        self.status_lbl.setText(f"Done → {doc_dir}")
        self.log(f"[process] done -> {doc_dir}")
        QMessageBox.information(self, "Completed", f"Saved results to:\n{doc_dir}")

    def _an_process_cleanup(self):
        self.proc_thread = None
        self.proc_worker = None
        self.btn_process.setEnabled(True)

    # ---------- shared error handler ----------
    def _on_worker_failed(self, err: str):
        if self._an_watchdog:
            self._an_watchdog.stop()
        self.status_lbl.setText("Failed")
        self.log("[error]\n" + err)
        QMessageBox.critical(self, "Error", err)

    # ---------- helpers ----------
    def _selected_lang_keys(self) -> Set[str]:
        s: Set[str] = set()
        if self.chk_en.isChecked():
            s.add("english")
        if self.chk_jp_mod.isChecked():
            s.add("jp_modern")
        if self.chk_jp_cls.isChecked():
            s.add("jp_classical")
        return s

    def _update_process_enabled(self):
        self.btn_process.setEnabled(self.current_image is not None and len(self._selected_lang_keys()) > 0)


if __name__ == "__main__":
    # Configure basic logging before creating the app
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

