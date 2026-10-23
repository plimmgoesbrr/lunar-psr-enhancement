"""
Stage 1 — Data Mocking
======================
Generates synthetic "sunlit crater" base images via procedural generation.

Real inputs for the final system are ISRO Chandrayaan-2 OHRC frames and
NASA LRO (LROC NAC + LOLA DEM) products. Those are multi-GB radiometric
products, so for a quick demo this module fabricates realistic stand-ins:

  fractal value noise (Perlin-style)  ->  regolith texture
  + parametric crater bowls with directional-lit rims
  + scattered boulders
  = plausible sunlit crater scene in [0, 1]

Usage:
    python -m src.data_mock --n 8 --size 256 --out data/sunlit
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

RNG = np.random.default_rng


# --------------------------------------------------------------------------
# Perlin-style fractal value noise (numpy-only, no external deps)
# --------------------------------------------------------------------------
def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def value_noise(size: int, grid: int, rng) -> np.ndarray:
    """One octave of bilinearly-interpolated lattice noise, size x size."""
    lattice = rng.random((grid + 1, grid + 1)).astype(np.float32)
    xs = np.linspace(0, grid, size, endpoint=False)
    x0 = xs.astype(int)
    tx = _smoothstep((xs - x0)[None, :])
    ty = _smoothstep((xs - x0)[:, None])
    a = lattice[np.ix_(x0, x0)]
    b = lattice[np.ix_(x0, x0 + 1)]
    c = lattice[np.ix_(x0 + 1, x0)]
    d = lattice[np.ix_(x0 + 1, x0 + 1)]
    top = a * (1 - tx) + b * tx
    bot = c * (1 - tx) + d * tx
    return top * (1 - ty) + bot * ty


def fractal_noise(size: int, octaves: int = 5, rng=None) -> np.ndarray:
    """Sum of octaves -> Perlin-style fractal terrain texture in [0,1]."""
    rng = rng or RNG()
    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        grid = max(2, 2 ** (o + 2))
        out += amp * value_noise(size, grid, rng)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (np.ptp(out) + 1e-8)


# --------------------------------------------------------------------------
# Crater synthesis with directional illumination
# --------------------------------------------------------------------------
def add_crater(img: np.ndarray, cx: float, cy: float, radius: float,
               depth: float, sun_dir: float, rng) -> None:
    """Stamp one crater: darkened bowl + sun-facing bright rim (in place)."""
    size = img.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    d = np.sqrt(dx * dx + dy * dy)

    # bowl: cosine falloff toward center
    bowl = np.clip(1 - d / radius, 0, 1)
    img -= depth * np.sin(bowl * np.pi / 2) * 0.6

    # rim: narrow gaussian ring, brighter on the sun-facing side
    ring = np.exp(-((d - radius) / (radius * 0.08 + 1.5)) ** 2)
    azimuth = np.arctan2(dy, dx)
    lit = 0.5 + 0.5 * np.cos(azimuth - sun_dir)          # 1 facing sun
    img += ring * (0.15 + 0.45 * lit) * depth

    # inner shadow opposite the sun
    inner = np.exp(-((d - radius * 0.55) / (radius * 0.25)) ** 2)
    shade = 0.5 + 0.5 * np.cos(azimuth - sun_dir - np.pi)
    img -= inner * shade * depth * 0.35


def add_boulders(img: np.ndarray, n: int, rng) -> None:
    size = img.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for _ in range(n):
        bx, by = rng.uniform(0, size, 2)
        r = rng.uniform(1.5, 4.0)
        blob = np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * r * r))
        img += blob * rng.uniform(0.25, 0.55)


def make_sunlit_crater(size: int = 256, seed: int | None = None) -> np.ndarray:
    """One synthetic sunlit crater scene, float32 in [0, 1]."""
    rng = RNG(seed)
    img = 0.35 + 0.45 * fractal_noise(size, octaves=5, rng=rng)
    sun_dir = rng.uniform(0, 2 * np.pi)

    n_craters = rng.integers(2, 5)
    for _ in range(n_craters):
        radius = rng.uniform(size * 0.10, size * 0.30)
        add_crater(img,
                   cx=rng.uniform(size * 0.2, size * 0.8),
                   cy=rng.uniform(size * 0.2, size * 0.8),
                   radius=radius,
                   depth=rng.uniform(0.25, 0.5),
                   sun_dir=sun_dir, rng=rng)

    add_boulders(img, n=int(rng.integers(10, 25)), rng=rng)
    return np.clip(img, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic sunlit craters")
    ap.add_argument("--n", type=int, default=8, help="number of images")
    ap.add_argument("--size", type=int, default=256, help="image size (px)")
    ap.add_argument("--out", type=str, default="data/sunlit", help="output dir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import cv2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        img = make_sunlit_crater(args.size, seed=args.seed + i)
        cv2.imwrite(str(out / f"sunlit_{i:03d}.png"),
                    (img * 65535).astype(np.uint16))   # 16-bit like real products
    print(f"[data_mock] wrote {args.n} x {args.size}px sunlit craters -> {out}")


if __name__ == "__main__":
    main()
