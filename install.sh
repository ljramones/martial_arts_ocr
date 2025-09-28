#!/usr/bin/env bash
set -euo pipefail

# Defaults
VENV_DIR=".venv"
CORE_LOCK="requirements-lock-core.txt"
ARGOS_EXTRAS="extras-argos.txt"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CREATE_VENV=0

usage() {
  cat <<USAGE
Usage: $0 [--create-venv] [--venv DIR] [--python PATH]

Installs:
  1) Core stack from $CORE_LOCK
  2) Argos stack from $ARGOS_EXTRAS with --no-deps (keeps sentencepiece==0.2.0)

Options:
  --create-venv        Create a venv at \$PWD/$VENV_DIR (if missing) and use it
  --venv DIR           Venv directory (default: $VENV_DIR)
  --python PATH        Python interpreter to use (default: $PYTHON_BIN)
  -h, --help           Show this help

Env vars:
  PYTHON_BIN           Override Python interpreter path
USAGE
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --create-venv) CREATE_VENV=1; shift ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

# Checks
[[ -f "$CORE_LOCK" ]] || { echo "Missing $CORE_LOCK. Generate it first."; exit 1; }
[[ -f "$ARGOS_EXTRAS" ]] || { echo "Missing $ARGOS_EXTRAS. Create it first."; exit 1; }

# Create/activate venv if requested
if [[ $CREATE_VENV -eq 1 && ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate venv if it exists but not active
if [[ -d "$VENV_DIR" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

# Confirm python
echo "Using Python: $(command -v python)"
python -c 'import platform,struct; print("Python", platform.python_version(), platform.machine(), str(struct.calcsize("P")*8)+"-bit") )'

# Upgrade tooling
python -m pip install -U pip setuptools wheel

# Install core (fully pinned)
echo "Installing core from $CORE_LOCK ..."
pip install -r "$CORE_LOCK"

# Install Argos WITHOUT deps to keep sentencepiece==0.2.0
echo "Installing Argos from $ARGOS_EXTRAS (no-deps) ..."
pip install --no-deps -r "$ARGOS_EXTRAS"

# Final report
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
import platform, sentencepiece
def v(name):
    try: return version(name)
    except PackageNotFoundError: return "not-installed"
print("== Environment ==")
print("Python:", platform.python_version(), platform.machine())
for pkg in ["Flask","easyocr","opencv-python","opencv-python-headless","numpy","scikit-image",
            "torch","torchvision","argostranslate","sentencepiece"]:
    try:
        print(f"{pkg:22s}", v(pkg))
    except Exception as e:
        print(f"{pkg:22s}", f"error: {e}")
PY

echo "Done."
