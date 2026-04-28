import os
os.environ["QT_PLUGIN_PATH"] = ""
os.environ["QT_MEDIA_BACKEND"] = "none"
os.environ["QT_QPA_PLATFORM"] = "cocoa"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_OPENGL"] = "software"
os.environ["QT_MAC_WANTS_LAYER"] = "1"

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, QCheckBox, QPlainTextEdit
from PySide6.QtCore import Qt

app = QApplication([])
win = QMainWindow()
win.setWindowTitle("Smoke UI")

splitter = QSplitter()
left = QWidget(); mid = QWidget(); right = QWidget()
left_l = QVBoxLayout(left); mid_l = QVBoxLayout(mid); right_l = QVBoxLayout(right)

dir_label = QLabel("No folder")
choose = QPushButton("Choose")
files = QListWidget()
left_l.addWidget(dir_label); left_l.addWidget(choose); left_l.addWidget(files)

preview = QLabel("Preview"); preview.setAlignment(Qt.AlignCenter)
recognize = QPushButton("Recognize")
process = QPushButton("Process")
mid_btns = QHBoxLayout(); mid_btns.addWidget(recognize); mid_btns.addWidget(process)
mid_l.addWidget(preview); mid_l.addLayout(mid_btns)

en = QCheckBox("English"); jm = QCheckBox("JP Modern"); jc = QCheckBox("JP Classical")
log = QPlainTextEdit(); log.setReadOnly(True)
right_l.addWidget(en); right_l.addWidget(jm); right_l.addWidget(jc); right_l.addWidget(log)

splitter.addWidget(left); splitter.addWidget(mid); splitter.addWidget(right)
win.setCentralWidget(splitter)
win.resize(1200, 800); win.show()
print("UI OK")
app.exec()
