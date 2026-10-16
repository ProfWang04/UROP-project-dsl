#!/bin/bash
# CSE COMPUTE job for Project DSL hard L1 grid (4 GPUs).
# Workdir: /project/zihanz_project1/urop-dsl-hard-l1
# SUBMIT JOB: GPU count = 4 (or 1 for smoke/diag)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
# Force line-buffered Python so logs appear while training (not only on crash/exit).
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

mkdir -p results/grid_kN logs

# Append to both per-cell log and COMPUTE job stdout (tee -a).
log_line() {
  local log="$1"; shift
  local msg="[$(date -Is)] $*"
  printf '%s\n' "$msg" | tee -a "$log"
}

cuda_diag() {
  echo "===== CUDA diag $(date -Is) ====="
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
  command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi: missing"
  python3 - <<'PY'
import os, sys
print("python", sys.version.split()[0])
try:
    import torch
except Exception as e:
    print("torch import failed:", e)
    raise SystemExit(1)
print("torch", torch.__version__, "cuda_built", torch.version.cuda)
print("cuda.is_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  gpu[{i}]", torch.cuda.get_device_name(i))
    x = torch.zeros(1, device="cuda")
    print("cuda smoke tensor ok", x.device)
else:
    print("CUDA NOT AVAILABLE — training with --device cuda will fail")
    raise SystemExit(2)
PY
}

run_one() {
  local gpu="$1" k="$2" n="$3" seed="$4" arm="${5:-hard}"
  local out="results/grid_kN/k${k}_n${n}_s${seed}_${arm}"
  local log="logs/k${k}_n${n}_s${seed}_${arm}.log"
  : >"$log"  # truncate so this run is readable from line 1
  log_line "$log" "start gpu=$gpu k=$k N=$n seed=$seed arm=$arm -> $out"
  log_line "$log" "env CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
  # train.py copies the full train split to $out/train.jsonl and writes every
  # free-gen sample (not the 5-row history preview) to:
  #   $out/free_gen_test_iid.jsonl
  #   $out/free_gen_test_holdout.jsonl
  # Per-epoch copies are free_gen_<split>_epoch<E>.jsonl; history examples_path
  # points at those. --dump-free-gen is the default; passed here so grid logs show it.
  log_line "$log" "dumps: $out/train.jsonl $out/free_gen_test_iid.jsonl $out/free_gen_test_holdout.jsonl"
  # Remap so this process only sees one GPU as cuda:0 (avoids multi-proc CUDA fights).
  if CUDA_VISIBLE_DEVICES="$gpu" python3 -u -m src.train \
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
    --dump-free-gen \
    --out "$out" \
    > >(tee -a "$log") 2>&1
  then
    log_line "$log" "done gpu=$gpu k=$k N=$n seed=$seed -> $out"
  else
    local rc=$?
    log_line "$log" "FAILED rc=$rc gpu=$gpu k=$k N=$n seed=$seed"
    return "$rc"
  fi
}

case "${1:-grid4}" in
  selfcheck)
    python3 -m pytest tests/ -q
    ;;

  diag)
    cuda_diag
    ;;

  # First parallel cut on 4 GPUs: hard k=6 + gate knobs, N∈{1e3,1e4}, 2 seeds
  # (Current hard stack freezes HARD_API_K=6; k=8 wave comes after this.)
  grid4)
    cuda_diag | tee logs/cuda_diag_grid4.log
    run_one 0 6 1000 0 hard &
    p0=$!
    run_one 1 6 1000 1 hard &
    p1=$!
    run_one 2 6 10000 0 hard &
    p2=$!
    run_one 3 6 10000 1 hard &
    p3=$!
    fail=0
    for p in $p0 $p1 $p2 $p3; do
      if ! wait "$p"; then fail=1; fi
    done
    echo "grid4 finished (fail=$fail). See logs/ and results/grid_kN/"
    exit "$fail"
    ;;

  # Second seed wave for k=6 (still 4 GPUs)
  grid4-seeds23)
    cuda_diag | tee logs/cuda_diag_grid4_seeds23.log
    run_one 0 6 1000 2 hard &
    p0=$!
    run_one 1 6 1000 3 hard &
    p1=$!
    run_one 2 6 10000 2 hard &
    p2=$!
    run_one 3 6 10000 3 hard &
    p3=$!
    fail=0
    for p in $p0 $p1 $p2 $p3; do
      if ! wait "$p"; then fail=1; fi
    done
    exit "$fail"
    ;;

  # Single cell: bash job.sh cell <gpu> <n> <seed> [arm]
  cell)
    run_one "${2:?gpu}" 6 "${3:?n}" "${4:?seed}" "${5:-hard}"
    ;;

  # Smoke: one short hard+gate run on cuda:0 (SUBMIT JOB GPU count = 1)
  smoke)
    cuda_diag | tee logs/cuda_diag_smoke.log
    : >logs/smoke_k6_n200.log
    log_line logs/smoke_k6_n200.log "smoke start"
    log_line logs/smoke_k6_n200.log "dumps: results/grid_kN/smoke_k6_n200/train.jsonl results/grid_kN/smoke_k6_n200/free_gen_test_iid.jsonl results/grid_kN/smoke_k6_n200/free_gen_test_holdout.jsonl"
    CUDA_VISIBLE_DEVICES=0 python3 -u -m src.train \
      --hard --gate-knobs --arm hard \
      --n-train 200 --epochs 2 --batch-size 32 --block-size 128 \
      --device cuda --max-eval 20 \
      --dump-free-gen \
      --out results/grid_kN/smoke_k6_n200 \
      > >(tee -a logs/smoke_k6_n200.log) 2>&1
    log_line logs/smoke_k6_n200.log "smoke done"
    ;;

  *)
    echo "usage: bash job.sh selfcheck|diag|smoke|grid4|grid4-seeds23|cell <gpu> <n> <seed> [arm]" >&2
    exit 1
    ;;
esac
