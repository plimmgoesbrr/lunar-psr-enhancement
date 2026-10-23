"""
Model — compact U-Net for low-light enhancement
===============================================
Encoder-decoder with skip connections (Ronneberger et al., 2015), sized for
single-GPU training on image patches. The network predicts the *bright*
image directly from the *dark* input; skip connections carry fine spatial
detail (boulders, small craters) past the bottleneck so it survives
enhancement.

~1.9 M parameters at base=32 — trains comfortably on one consumer GPU
(and tolerably on CPU for smoke tests).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """In: (B,1,H,W) dark image in [0,1]. Out: (B,1,H,W) enhanced, [0,1]."""

    def __init__(self, base: int = 32):
        super().__init__()
        b = base
        self.enc1 = _block(1, b)
        self.enc2 = _block(b, b * 2)
        self.enc3 = _block(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)

        self.mid = _block(b * 4, b * 8)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = _block(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = _block(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = _block(b * 2, b)

        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = UNet()
    x = torch.rand(2, 1, 128, 128)
    y = net(x)
    print(f"UNet: {count_params(net)/1e6:.2f} M params | {tuple(x.shape)} -> {tuple(y.shape)}")
