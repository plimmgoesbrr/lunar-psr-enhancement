"""
Stage 4 — Inference
===================
Takes a raw low-light crater image and outputs a brightened, denoised image
using the trained U-Net checkpoint.

Handles 8-bit and 16-bit inputs. Images whose sides aren't multiples of 8
are reflection-padded so the 3-level U-Net downsampling works, then cropped
back to the original size.

Usage:
    python -m src.infer --ckpt checkpoints/unet.pt \
                        --in data/pairs/dark/sunlit_000.png \
                        --out outputs/enhanced_000.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import UNet


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"[infer] cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img /= 65535.0 if img.max() > 255 else 255.0
    return img


def enhance(img: np.ndarray, ckpt_path: str,
            device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    net = UNet().to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    h, w = img.shape
    ph, pw = (-h) % 8, (-w) % 8            # pad to multiple of 8
    x = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        y = net(t)[0, 0].cpu().numpy()
    return np.clip(y[:h, :w], 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Enhance a low-light crater image")
    ap.add_argument("--ckpt", type=str, default="checkpoints/unet.pt")
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    img = load_gray(args.inp)
    out = enhance(img, args.ckpt)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, (out * 65535).astype(np.uint16))
    print(f"[infer] {args.inp} -> {args.out}  "
          f"(in-range {img.min():.3f}-{img.max():.3f} -> out {out.min():.3f}-{out.max():.3f})")


if __name__ == "__main__":
    main()
