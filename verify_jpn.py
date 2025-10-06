# verify_jpn.py
from pathlib import Path
import os, pytesseract
from PIL import Image

REPO = Path(__file__).resolve().parent
TESSDATA = REPO / "tessdata"
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
os.environ["TESSDATA_PREFIX"] = str(TESSDATA)

IMG = REPO / "modern_japanese" / "JapText2.jpg"   # <-- change to a real JP file
img = Image.open(IMG).convert("L")                # grayscale is fine

cfg = f'--psm 6 --oem 1 --tessdata-dir "{TESSDATA}" -c preserve_interword_spaces=1'
txt = pytesseract.image_to_string(img, lang="jpn", config=cfg)
print("LEN:", len(txt))
print("SNIP:", txt.strip()[:80])
