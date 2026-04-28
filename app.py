"""Legacy launcher/import alias for :mod:`martial_arts_ocr.app.flask_app`."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from martial_arts_ocr.app import flask_app as _flask_app

if __name__ == "__main__":
    _flask_app.main()
else:
    sys.modules[__name__] = _flask_app
