#!/usr/bin/env bash
# Lightweight test runner for the package.
# Steps:
# 1. Ensure conda is initialised (mamba/conda on PATH).  
# 2. Create a local environment in ./environment using `conda create --prefix`.
# 3. Install requirements from package/requirements.txt (if present).
# 4. Run test script package_test/test_run.py (or a provided script).
# 5. Cleanup: remove the environment and any temporary files (unless KEEP_ENV=1).

# Set strict modes and internal field separator
set -euo pipefail
IFS=$'\n\t'

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_DIR="$ROOT_DIR/tests/package_test"
PACKAGE_DIR="$ROOT_DIR/package"
ENV_DIR="$ROOT_DIR/environment"
REQ_FILE="$PACKAGE_DIR/requirements.txt"
TEST_SCRIPT="$TEST_DIR/test_run.py"

KEEP_ENV=${KEEP_ENV:-0}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
CONDA_CMD=${CONDA_CMD:-conda}

function error_exit() {
  echo "ERROR: $1" >&2
  exit 1
}

echo "[*] Running package test from: $ROOT_DIR"

# Step 1) Check conda is available
if ! command -v "$CONDA_CMD" >/dev/null 2>&1; then
  error_exit "Conda command '$CONDA_CMD' not found. Make sure conda is installed and on PATH."
fi

# Step 2) Create environment folder
if [ -d "$ENV_DIR" ]; then
  echo "[*] Removing existing environment folder: $ENV_DIR"
  rm -rf "$ENV_DIR"
fi

echo "[*] Creating conda environment at: $ENV_DIR"
$CONDA_CMD create --prefix "$ENV_DIR" -y python=$PYTHON_VERSION pip || error_exit "Failed to create conda env"

# 3) Activate and install requirements
# Activation in non-interactive shells: use conda run or source the activate script
# Using `conda run` keeps things simpler for scripts (conda 4.6+)
if [ -f "$REQ_FILE" ]; then
  echo "[*] Installing requirements from $REQ_FILE"
  # Use pip inside the environment
  "$CONDA_CMD" run --prefix "$ENV_DIR" pip install -r "$REQ_FILE" || error_exit "pip install failed"
else
  echo "[*] No requirements.txt found at $REQ_FILE — skipping pip install"
fi

# Also ensure our package is available: install package/ in editable mode
if [ -f "$PACKAGE_DIR/setup.py" ]; then
  echo "[*] Installing package from $PACKAGE_DIR into environment"
  "$CONDA_CMD" run --prefix "$ENV_DIR" pip install -e "$PACKAGE_DIR" || error_exit "pip install package failed"
else
  echo "[*] No setup.py in $PACKAGE_DIR — skipping package install"
fi

# 4) Run the Python test script
if [ -f "$TEST_SCRIPT" ]; then
  echo "[*] Running test script: $TEST_SCRIPT"
  "$CONDA_CMD" run --prefix "$ENV_DIR" python "$TEST_SCRIPT"
  TEST_RESULT=$?
  if [ "$TEST_RESULT" -ne 0 ]; then
    echo "[!] Test script failed with exit code $TEST_RESULT"
  else
    echo "[*] Test script completed successfully"
  fi
else
  echo "[*] No test script found at $TEST_SCRIPT — nothing to run"
fi

# 5) Cleanup
if [ "$KEEP_ENV" -eq 1 ]; then
  echo "[*] KEEP_ENV=1 — skipping environment removal: $ENV_DIR"
else
  echo "[*] Removing environment folder: $ENV_DIR"
  rm -rf "$ENV_DIR"
fi

echo "[*] Done"
