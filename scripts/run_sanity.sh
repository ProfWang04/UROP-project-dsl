#!/usr/bin/env bash
# CPU sanity: L1 k=3, N=1000 NTP
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pytest tests/ -q
python3 -m src.train \
  --level l1 \
  --api-k 3 \
  --n-train 1000 \
  --epochs 15 \
  --batch-size 32 \
  --out results/runs/l1_k3_n1000 \
  --eval-every 5 \
  --max-eval 50 \
  --device cpu
