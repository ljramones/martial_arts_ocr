from pathlib import Path
import os, subprocess
import pytesseract
from PIL import Image, ImageDraw

# Absolute paths
REPO = Path(__file__).resolve().parent
TESSDATA = REPO / "tessdata"
TESSERACT = "/opt/homebrew/bin/tesseract"  # brew path on Apple Silicon

# Tell pytesseract which binary to use
pytesseract.pytesseract.tesseract_cmd = TESSERACT

# Option A: point env var directly to the tessdata directory
os.environ["TESSDATA_PREFIX"] = str(TESSDATA)  # <-- note: points to tessdata, not its parent

# Option B (also works): pass --tessdata-dir via config (uncomment to use)
# CONFIG = f'--psm 6 --tessdata-dir "{TESSDATA}"'
CONFIG = "--psm 6"  # we'll rely on TESSDATA_PREFIX above

print("tesseract:", subprocess.check_output([TESSERACT, "--version"]).decode().splitlines()[0])
print("TESSDATA_PREFIX:", os.environ["TESSDATA_PREFIX"])
print("langs:", pytesseract.get_languages(config=f'--tessdata-dir "{TESSDATA}"'))

# Create a tiny JP image and OCR it
img = Image.new("L", (600, 120), 255)
ImageDraw.Draw(img).text((10, 40), "日本語のテスト", fill=0)
try:
    txt = pytesseract.image_to_string(img, lang="eng+jpn", config=CONFIG)
except pytesseract.TesseractError:
    txt = pytesseract.image_to_string(img, lang="jpn", config=CONFIG)

print("OCR:", txt.strip())
