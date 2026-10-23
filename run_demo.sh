#!/usr/bin/env bash
# End-to-end demo: mock data -> degrade -> train -> infer -> validate
# Quick settings so it completes in minutes even on CPU.
set -euo pipefail

STEPS=${STEPS:-150}
PATCH=${PATCH:-96}
BATCH=${BATCH:-4}

echo "== Stage 1: synthetic sunlit craters =="
python -m src.data_mock --n 6 --size 256 --out data/sunlit

echo "== Stage 2: physics-based degradation -> (dark, bright) pairs =="
python -m src.degrade --in data/sunlit --out data/pairs

echo "== Stage 3: train U-Net (steps=$STEPS, patch=$PATCH, batch=$BATCH) =="
python -m src.train --steps "$STEPS" --batch "$BATCH" --patch "$PATCH" --out checkpoints

echo "== Stage 4: inference on a held-out dark frame =="
python -m src.infer --ckpt checkpoints/unet.pt \
                    --in data/pairs/dark/sunlit_000.png \
                    --out outputs/enhanced_000.png

echo "== Stage 5: validation (PSNR / SSIM) =="
python -m src.validate --pred outputs/enhanced_000.png \
                       --target data/pairs/bright/sunlit_000.png \
                       --baseline data/pairs/dark/sunlit_000.png

echo "== demo complete: outputs/enhanced_000.png =="
