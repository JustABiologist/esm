#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ESMFOLD2_PYTHON:-$ROOT/.conda-esmfold2/bin/python}"
MODEL_DIR="${ESMFOLD2_MODEL_ID:-$ROOT/.hf-models/biohub_ESMFold2}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ESMFold2 Python environment: $PYTHON" >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
  echo "Missing local ESMFold2 model snapshot: $MODEL_DIR" >&2
  exit 1
fi

export ESMFOLD2_RUN_INFERENCE="${ESMFOLD2_RUN_INFERENCE:-1}"
export ESMFOLD2_MODEL_ID="$MODEL_DIR"
export ESMFOLD2_CCD_CACHE="${ESMFOLD2_CCD_CACHE:-$MODEL_DIR}"
export ESMCFOLD_CCD_PATH="${ESMCFOLD_CCD_PATH:-$MODEL_DIR/ccd.pkl}"
export ESMFOLD2_DEVICE="${ESMFOLD2_DEVICE:-cpu}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

"$PYTHON" -m pytest -q --no-cov \
  "$ROOT/esm/models/esmfold2/pocket_conditioning_test.py" "$@"
