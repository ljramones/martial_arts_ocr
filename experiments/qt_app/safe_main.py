# qt_app/safe_main.py
# --- Hardening ENV before Qt loads ---
import os
os.environ.setdefault("QT_PLUGIN_PATH", "")
os.environ.setdefault("QT_MEDIA_BACKEND", "none")
os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
os.environ.setdefault("QT_OPENGL", "software")

import sys
from pathlib import Path
from typing import List, Set

from PySide6.QtCore import Qt, QObject, QThread, Signal, QTimer, QCoreApplication
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QCheckBox,
    QMessageBox, QSplitter, QPlainTextEdit, QSizePolicy
)

# Avoid any Qt imageformat plugins
def _prune_qt_imageformat_plugins():
    paths = [p for p in QCoreApplication.libraryPaths() if "imageformats" not in p.lower()]
    QCoreApplication.setLibraryPaths(paths)

# Pillow preview to bypass Qt plugins entirely
from PIL import Image, ImageOps

# ✅ Use the new unified orchestrator
from experiments.qt_app.orchestrator import Orchestrator, IMG_EXTS


# ---------- Workers ----------
class AnalyzeWorker(QObject):
    step = Signal(str)
    finished = Signal(list)   # list[str]
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


# ---------- Main Window (crash-safe) ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Martial Arts OCR — Crash-Safe UI")
        self.resize(1300, 850)

        self.orch = Orchestrator(processed_root="data/processed")

        splitter = QSplitter(self)
        left = QWidget(); mid = QWidget(); right = QWidget()
        left_l = QVBoxLayout(left); mid_l = QVBoxLayout(mid); right_l = QVBoxLayout(right)

        # Left: folder + files
        self.dir_label = QLabel("No folder chosen")
        self.btn_choose_dir = QPushButton("Choose Source Folder…")
        self.files_list = QListWidget()
        left_l.addWidget(self.dir_label)
        left_l.addWidget(self.btn_choose_dir)
        left_l.addWidget(self.files_list)

        # Middle: image preview (QLabel, not QGraphicsView)
        self.preview = QLabel("Select an image…")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setMinimumSize(400, 400)

        mid_buttons = QHBoxLayout()
        self.btn_recognize = QPushButton("Recognize Types")
        self.btn_process = QPushButton("Process")
        self.btn_process.setEnabled(False)
        mid_buttons.addWidget(self.btn_recognize)
        mid_buttons.addWidget(self.btn_process)

        mid_l.addWidget(self.preview, stretch=1)
        mid_l.addLayout(mid_buttons)

        # Right: checkboxes + log
        self.chk_en = QCheckBox("English")
        self.chk_jp_mod = QCheckBox("Japanese (Modern)")
        self.chk_jp_cls = QCheckBox("Japanese (Classical)")
        for cb in (self.chk_en, self.chk_jp_mod, self.chk_jp_cls):
            cb.stateChanged.connect(self._update_process_enabled)

        self.log_out = QPlainTextEdit()
        self.log_out.setReadOnly(True)
        self.log_out.setMaximumBlockCount(2000)

        self.status_lbl = QLabel("Ready.")

        right_l.addWidget(QLabel("Detected / Override (multiple allowed):"))
        right_l.addWidget(self.chk_en)
        right_l.addWidget(self.chk_jp_mod)
        right_l.addWidget(self.chk_jp_cls)
        right_l.addSpacing(10)
        right_l.addWidget(QLabel("Log"))
        right_l.addWidget(self.log_out, stretch=1)
        right_l.addSpacing(10)
        right_l.addWidget(self.status_lbl)

        splitter.addWidget(left); splitter.addWidget(mid); splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)

        # State
        self.current_dir: Path | None = None
        self.current_image: Path | None = None
        self.worker_thread: QThread | None = None
        self._an_watchdog: QTimer | None = None

        # Signals
        self.btn_choose_dir.clicked.connect(self.choose_dir)
        self.files_list.itemSelectionChanged.connect(self.on_file_selected)
        self.btn_recognize.clicked.connect(self.on_recognize_clicked)
        self.btn_process.clicked.connect(self.on_process_clicked)

    def log(self, msg: str): self.log_out.appendPlainText(msg)

    # Dir + list
    def choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select source folder", str(Path.home()))
        if not d: return
        self.current_dir = Path(d)
        self.dir_label.setText(str(self.current_dir))
        self.populate_files_list()

    def populate_files_list(self):
        self.files_list.clear(); cnt = 0
        if self.current_dir:
            for p in sorted(self.current_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.files_list.addItem(str(p)); cnt += 1
        self.status_lbl.setText(f"Found {cnt} image(s).")
        self.log(f"[list] {cnt} image(s) from {self.current_dir}")

    # Selection + preview (Pillow)
    def on_file_selected(self):
        for cb in (self.chk_en, self.chk_jp_mod, self.chk_jp_cls):
            cb.blockSignals(True); cb.setChecked(False); cb.blockSignals(False)
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
                qimg = QImage(im.tobytes("raw", "RGBA"), im.width, im.height, QImage.Format_RGBA8888)
            pix = QPixmap.fromImage(qimg)
            self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.log(f"[preview] error: {e}")
            self.preview.setText(f"Preview error:\n{e}")

    # Recognize (multi-label)
    def on_recognize_clicked(self):
        if not self.current_image:
            QMessageBox.information(self, "No file", "Select an image from the list.")
            return
        self.status_lbl.setText("Recognizing types…")
        self.log("=== Recognize Types ==="); self.log(f"File: {self.current_image}")

        self.worker_thread = QThread()
        worker = AnalyzeWorker(self.orch, str(self.current_image))
        worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(worker.run)
        worker.step.connect(lambda m: self.log(f"[analyze] {m}"))
        worker.finished.connect(self._on_analyze_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(self.worker_thread.quit)
        worker.failed.connect(self.worker_thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

        self._an_watchdog = QTimer(self); self._an_watchdog.setSingleShot(True)
        self._an_watchdog.timeout.connect(self._on_analyze_watchdog)
        self._an_watchdog.start(15000)

    def _on_analyze_watchdog(self):
        self.log("[watchdog] analyze still running after 15s…")
        self.status_lbl.setText("Analyzer taking a while…")

    def _on_analyze_finished(self, lang_keys: List[str]):
        if self._an_watchdog: self._an_watchdog.stop()
        self.chk_en.setChecked("english" in lang_keys)
        self.chk_jp_mod.setChecked("jp_modern" in lang_keys)
        self.chk_jp_cls.setChecked("jp_classical" in lang_keys)
        self.status_lbl.setText(f"Recognized: {', '.join(lang_keys) if lang_keys else 'none'}")
        self._update_process_enabled()
        self.log(f"[analyze] done -> {lang_keys}")

    # Process
    def on_process_clicked(self):
        if not self.current_image:
            QMessageBox.information(self, "No file", "Select an image first.")
            return
        selected = self._selected_lang_keys()
        if not selected:
            QMessageBox.information(self, "Choose types", "Pick at least one language box.")
            return
        self.status_lbl.setText(f"Processing as: {', '.join(sorted(selected))}…")
        self.log("=== Process ==="); self.log(f"File: {self.current_image}"); self.log(f"Langs: {sorted(selected)}")

        self.worker_thread = QThread()
        worker = ProcessWorker(self.orch, str(self.current_image), selected)
        worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(worker.run)
        worker.progress.connect(lambda pct, msg: self.log(f"[process] {pct}% {msg}"))
        worker.finished.connect(self._on_process_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(self.worker_thread.quit)
        worker.failed.connect(self.worker_thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _on_process_finished(self, doc_dir: str):
        self.status_lbl.setText(f"Done → {doc_dir}")
        self.log(f"[process] done -> {doc_dir}")
        QMessageBox.information(self, "Completed", f"Saved results to:\n{doc_dir}")

    def _on_worker_failed(self, err: str):
        if self._an_watchdog: self._an_watchdog.stop()
        self.status_lbl.setText("Failed")
        self.log("[error]\n" + err)
        QMessageBox.critical(self, "Error", err)

    def _selected_lang_keys(self) -> Set[str]:
        s: Set[str] = set()
        if self.chk_en.isChecked(): s.add("english")
        if self.chk_jp_mod.isChecked(): s.add("jp_modern")
        if self.chk_jp_cls.isChecked(): s.add("jp_classical")
        return s

    def _update_process_enabled(self):
        self.btn_process.setEnabled(self.current_image is not None and len(self._selected_lang_keys()) > 0)


# ---------- Entrypoint ----------
if __name__ == "__main__":
    _prune_qt_imageformat_plugins()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
