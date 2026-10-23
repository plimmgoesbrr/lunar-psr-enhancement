"""
Stage 2 — Physics-Based Degradation
===================================
Turns a *sunlit* crater image into a realistic *PSR-like* low-light capture,
producing the (dark input, bright target) pairs used for training.

Camera physics modelled
-----------------------
1. Illumination scaling      I_dark = k * I_sunlit          (k ~ 1-8 %)
   PSR floors receive only faint secondary illumination scattered off
   sunlit crater walls, modelled as a weak, spatially-smooth gradient.
2. Photon shot noise         Poisson(I * ppp) / ppp
   At low signal the discrete-photon nature of light dominates: variance
   scales with the signal itself (signal-dependent noise).
3. Sensor read noise         + N(0, sigma_r)
   Electronics add signal-independent Gaussian noise on readout.
4. Quantization              round to the ADC bit depth.

Together: the standard Poisson-Gaussian sensor model used in
low-light imaging literature (cf. HORUS, Moseley et al., CVPR 2021).

Usage:
    python -m src.degrade --in data/sunlit --out data/pairs
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RNG = np.random.default_rng


@dataclass
class DegradeParams:
    k_min: float = 0.01          # min illumination fraction (1 %)
    k_max: float = 0.08          # max illumination fraction (8 %)
    photons_per_pixel: float = 40.0   # full-scale photon budget (shot noise)
    read_sigma: float = 0.008    # read-noise std, in [0,1] units
    secondary_grad: float = 0.006     # max scattered-light gradient
    bits: int = 12               # ADC bit depth (OHRC-like)


def degrade(sunlit: np.ndarray, params: DegradeParams = DegradeParams(),
            rng=None) -> np.ndarray:
    """sunlit float32 [0,1]  ->  dark noisy float32 [0,1] (same shape)."""
    rng = rng or RNG()
    h, w = sunlit.shape

    # 1) illumination scaling + smooth secondary-illumination gradient
    k = rng.uniform(params.k_min, params.k_max)
    theta = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (np.cos(theta) * xx / w + np.sin(theta) * yy / h)
    ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-8)
    dark = sunlit * k + ramp * params.secondary_grad

    # 2) photon shot noise (Poisson, signal-dependent)
    ppp = params.photons_per_pixel
    photons = rng.poisson(np.clip(dark, 0, 1) * ppp).astype(np.float32)
    dark = photons / ppp

    # 3) sensor read noise (Gaussian, signal-independent)
    dark += rng.normal(0.0, params.read_sigma, size=dark.shape).astype(np.float32)

    # 4) quantization to ADC levels
    levels = 2 ** params.bits - 1
    dark = np.round(np.clip(dark, 0, 1) * levels) / levels
    return dark.astype(np.float32)


def make_pair(sunlit: np.ndarray, params: DegradeParams = DegradeParams(),
              rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (dark_input, bright_target)."""
    return degrade(sunlit, params, rng), sunlit


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize (dark, bright) pairs")
    ap.add_argument("--in", dest="inp", type=str, default="data/sunlit")
    ap.add_argument("--out", type=str, default="data/pairs")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import cv2
    rng = RNG(args.seed)
    inp, out = Path(args.inp), Path(args.out)
    (out / "dark").mkdir(parents=True, exist_ok=True)
    (out / "bright").mkdir(parents=True, exist_ok=True)

    files = sorted(inp.glob("*.png"))
    if not files:
        raise SystemExit(f"[degrade] no PNGs found in {inp} - run data_mock first")

    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32)
        img /= 65535.0 if img.max() > 255 else 255.0
        dark, bright = make_pair(img, rng=rng)
        cv2.imwrite(str(out / "dark" / f.name), (dark * 65535).astype(np.uint16))
        cv2.imwrite(str(out / "bright" / f.name), (bright * 65535).astype(np.uint16))
    print(f"[degrade] wrote {len(files)} (dark, bright) pairs -> {out}")


if __name__ == "__main__":
    main()
