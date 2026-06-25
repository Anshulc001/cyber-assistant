#!/usr/bin/env bash
# setup.sh — Bootstrap the Python environment for local development.
# Run from the repo root:
#   bash scripts/setup.sh

set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo "==> Checking Python version …"
$PYTHON --version

echo "==> Creating virtual environment in '$VENV_DIR' …"
$PYTHON -m venv "$VENV_DIR"

echo "==> Activating virtual environment …"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate"

echo "==> Upgrading pip …"
pip install --quiet --upgrade pip

echo "==> Installing dependencies from requirements.txt …"
pip install --quiet -r requirements.txt

echo ""
echo "✅  Setup complete."
echo ""
echo "To activate the environment in future sessions:"
echo "  source $VENV_DIR/bin/activate       # Linux / macOS"
echo "  $VENV_DIR\\Scripts\\activate          # Windows PowerShell"
echo ""
echo "To start the API server locally:"
echo "  uvicorn backend.main:app --reload"
echo "  # then open http://127.0.0.1:8000/health"
