"""Compatibility wrapper for :mod:`martial_arts_ocr.db.models`."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from martial_arts_ocr.db.models import *  # noqa: F401,F403
