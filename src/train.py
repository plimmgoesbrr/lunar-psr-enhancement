"""
Stage 3 — Model Training
========================
Trains the U-Net on (dark, bright) pairs synthesized *on the fly*:
each sample is a fresh procedural sunlit crater (stage 1) pushed through
the physics-based degradation model (stage 2). Infinite training data,
zero disk footprint.

Runs on a single GPU when available, falls back to CPU for smoke tests.

Loss:      L1 (robust for restoration; sharper than L2)
Optimizer: Adam
Output:    checkpoints/unet.pt  + loss log

Usage:
    python -m src.train --steps 2000 --batch 8 --patch 128
    python -m src.train --steps 30 --batch 2 --patch 64      # CPU smoke test
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .data_mock import make_sunlit_crater
from .degrade import DegradeParams, degrade
from .model import UNet, count_params


class SyntheticPairs(Dataset):
    """Infinite-style dataset: procedural crater -> physics degradation."""

    def __init__(self, patch: int = 128, length: int = 10_000,
                 params: DegradeParams | None = None, seed: int = 0):
        self.patch = patch
        self.length = length
        self.params = params or DegradeParams()
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        bright = make_sunlit_crater(self.patch, seed=int(rng.integers(1 << 31)))
        dark = degrade(bright, self.params, rng=rng)
        to_t = lambda a: torch.from_numpy(a).unsqueeze(0)  # (1,H,W)
        return to_t(dark), to_t(bright)


def train(steps: int, batch: int, patch: int, lr: float,
          out_dir: str, workers: int = 2, log_every: int = 25) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device.type}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    net = UNet().to(device)
    print(f"[train] UNet {count_params(net)/1e6:.2f} M params")

    ds = SyntheticPairs(patch=patch, length=steps * batch)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers,
                    pin_memory=(device.type == "cuda"))

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "loss_log.csv"
    log = open(log_path, "w")
    log.write("step,loss\n")

    net.train()
    t0 = time.time()
    for step, (dark, bright) in enumerate(dl, 1):
        dark, bright = dark.to(device), bright.to(device)
        pred = net(dark)
        loss = loss_fn(pred, bright)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        log.write(f"{step},{loss.item():.6f}\n")
        if step % log_every == 0 or step == 1:
            ips = step * batch / (time.time() - t0)
            print(f"[train] step {step:>5}/{steps}  L1={loss.item():.4f}  ({ips:.1f} img/s)")
        if step >= steps:
            break
    log.close()

    ckpt = out / "unet.pt"
    torch.save({"model": net.state_dict(),
                "patch": patch, "steps": steps}, ckpt)
    print(f"[train] saved checkpoint -> {ckpt}")
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the enhancement U-Net")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    train(args.steps, args.batch, args.patch, args.lr, args.out, args.workers)


if __name__ == "__main__":
    main()
