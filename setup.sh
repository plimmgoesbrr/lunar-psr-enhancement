#!/usr/bin/env bash
# ---------------------------------------------------------------
# PSR-Pipeline environment setup
# Creates a Python virtualenv and installs the tech stack:
#   PyTorch · OpenCV · rasterio/GDAL · NumPy/SciPy
# Usage:  bash setup.sh          (CPU / default torch)
#         GPU=1 bash setup.sh    (CUDA 12.1 torch build)
# ---------------------------------------------------------------
set -euo pipefail

PY=${PYTHON:-python3}
VENV=.venv

echo "[setup] creating virtualenv at $VENV"
$PY -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip -q

if [[ "${GPU:-0}" == "1" ]]; then
  echo "[setup] installing CUDA-enabled PyTorch"
  pip install torch --index-url https://download.pytorch.org/whl/cu121
else
  echo "[setup] installing default PyTorch (CPU or system CUDA)"
  pip install torch
fi

echo "[setup] installing OpenCV, NumPy/SciPy, rasterio (GDAL bundled)"
pip install opencv-python-headless numpy scipy rasterio

echo "[setup] verifying imports"
python - << 'PYEOF'
import torch, cv2, numpy, scipy
print("  torch   ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("  opencv  ", cv2.__version__)
print("  numpy   ", numpy.__version__, "| scipy", scipy.__version__)
try:
    import rasterio
    print("  rasterio", rasterio.__version__, "(GDAL", rasterio.gdal_version() + ")")
except Exception as e:
    print("  rasterio: not available ->", e)
PYEOF

echo "[setup] done. Activate with:  source $VENV/bin/activate"
