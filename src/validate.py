"""
Stage 5 — Validation
====================
Rates enhanced images against the bright ground truth with the two standard
full-reference metrics:

  PSNR (dB)  — pixel-level fidelity; higher is better; >25 dB is decent,
               >30 dB good for restoration.
  SSIM       — perceived structural similarity in [0, 1]; compares local
               luminance / contrast / structure; 1.0 = identical.

Implemented with NumPy/SciPy only (gaussian-weighted SSIM per Wang et al.
2004), so no extra dependencies.

On *real* PSRs there is no bright ground truth — that is where no-reference
metrics (NIQE/BRISQUE) and LOLA-DEM consistency checks take over (future
work; see deck slide 4).

Usage:
    python -m src.validate --pred outputs/enhanced_000.png \
                           --target data/pairs/bright/sunlit_000.png \
                           [--baseline data/pairs/dark/sunlit_000.png]
"""
from __future__ import annotations
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter


# --------------------------------------------------------------------------
def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0,
         sigma: float = 1.5) -> float:
    """Gaussian-weighted SSIM (Wang et al., 2004), single-channel."""
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = gaussian_filter(x, sigma)
    mu_y = gaussian_filter(y, sigma)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    var_x = gaussian_filter(x * x, sigma) - mu_x2
    var_y = gaussian_filter(y * y, sigma) - mu_y2
    cov = gaussian_filter(x * y, sigma) - mu_xy

    num = (2 * mu_xy + c1) * (2 * cov + c2)
    den = (mu_x2 + mu_y2 + c1) * (var_x + var_y + c2)
    return float(np.mean(num / den))


def evaluate(pred: np.ndarray, target: np.ndarray) -> dict:
    return {"psnr_db": psnr(pred, target), "ssim": ssim(pred, target)}


# --------------------------------------------------------------------------
def _load(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"[validate] cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img /= 65535.0 if img.max() > 255 else 255.0
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="PSNR/SSIM validation")
    ap.add_argument("--pred", required=True, help="enhanced image")
    ap.add_argument("--target", required=True, help="bright ground truth")
    ap.add_argument("--baseline", help="optional raw dark input for comparison")
    args = ap.parse_args()

    pred, target = _load(args.pred), _load(args.target)
    m = evaluate(pred, target)
    print(f"[validate] enhanced vs truth : PSNR {m['psnr_db']:6.2f} dB | SSIM {m['ssim']:.4f}")

    if args.baseline:
        base = _load(args.baseline)
        b = evaluate(base, target)
        print(f"[validate] raw dark vs truth : PSNR {b['psnr_db']:6.2f} dB | SSIM {b['ssim']:.4f}")
        print(f"[validate] improvement       : +{m['psnr_db']-b['psnr_db']:.2f} dB | "
              f"+{m['ssim']-b['ssim']:.4f} SSIM")


if __name__ == "__main__":
    main()
