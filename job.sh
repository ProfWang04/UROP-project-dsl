#!/bin/bash
# CSE COMPUTE job for Project DSL hard L1 grid (4 GPUs).
# Workdir: /project/zihanz_project1/urop-dsl-hard-l1
# Image: stage1-distance  |  SUBMIT JOB: GPU count = 4
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

mkdir -p results/grid_kN logs

run_one() {
  local gpu="$1" k="$2" n="$3" seed="$4" arm="${5:-hard}"
  local out="results/grid_kN/k${k}_n${n}_s${seed}_${arm}"
  local log="logs/k${k}_n${n}_s${seed}_${arm}.log"
  echo "[$(date -Is)] start gpu=$gpu k=$k N=$n seed=$seed arm=$arm -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -m src.train \
    --hard \
    --gate-knobs \
    --arm "$arm" \
    --n-train "$n" \
    --seed "$seed" \
    --epochs 15 \
    --batch-size 32 \
    --block-size 128 \
    --device cuda \
    --eval-every 15 \
    --max-eval 80 \
    --out "$out" \
    >"$log" 2>&1
  echo "[$(date -Is)] done  gpu=$gpu k=$k N=$n seed=$seed -> $out"
}

case "${1:-grid4}" in
  selfcheck)
    python3 -m pytest tests/ -q
    ;;

  # First parallel cut on 4 GPUs: hard k=6 + gate knobs, N∈{1e3,1e4}, 2 seeds
  # (Current hard stack freezes HARD_API_K=6; k=8 wave comes after this.)
  grid4)
    run_one 0 6 1000 0 hard &
    run_one 1 6 1000 1 hard &
    run_one 2 6 10000 0 hard &
    run_one 3 6 10000 1 hard &
    wait
    echo "grid4 finished. Summaries under results/grid_kN/*/ and logs/"
    ;;

  # Second seed wave for k=6 (still 4 GPUs)
  grid4-seeds23)
    run_one 0 6 1000 2 hard &
    run_one 1 6 1000 3 hard &
    run_one 2 6 10000 2 hard &
    run_one 3 6 10000 3 hard &
    wait
    ;;

  # Single cell: bash job.sh cell <gpu> <n> <seed> [arm]
  cell)
    run_one "${2:?gpu}" 6 "${3:?n}" "${4:?seed}" "${5:-hard}"
    ;;

  # Smoke: one short hard+gate run on cuda:0
  smoke)
    CUDA_VISIBLE_DEVICES=0 python3 -m src.train \
      --hard --gate-knobs --arm hard \
      --n-train 200 --epochs 2 --batch-size 32 --block-size 128 \
      --device cuda --max-eval 20 \
      --out results/grid_kN/smoke_k6_n200
    ;;

  *)
    echo "usage: bash job.sh selfcheck|smoke|grid4|grid4-seeds23|cell <gpu> <n> <seed> [arm]" >&2
    exit 1
    ;;
esac
