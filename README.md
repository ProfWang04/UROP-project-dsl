# Project DSL — Phase 1

Executable mini-DSL with an **interpreter as hard verifier** and a small Transformer trained with **NTP only**. Goal: critical learnability vs API count `k` and data size `N`, with SCAN-style composition hold-out.

> **Repo status (2026-10-14):** code here is the **toy pipeline slice** (easy L1, `k=3`, `N=1000` sanity ≫ random). That run is a **historical baseline**, not the main claim. The active plan is the **harder L1 × full k×N×seed grid** — see [`docs/phase1-experiment-plan.md`](docs/phase1-experiment-plan.md) and the Project store plan.

Aligned with WJP × Zihan Zhang UROP explore (not UTM / not Dyck-only / no Phase-1 RL).

## Setup

```bash
python3 -m pip install -r requirements.txt
# or: pip install -e ".[dev]"
```

Requires Python ≥ 3.11 and PyTorch.

## Toy slice (pipeline proof)

```bash
PYTHONPATH=. python3 -m pytest tests/ -q

# Historical sanity — do not treat as main result
PYTHONPATH=. python3 -m src.train --level l1 --api-k 3 --n-train 1000 --epochs 15 \
  --out results/runs/l1_k3_n1000 --eval-every 5 --max-eval 50
```

Summary: `results/sanity_summary.json` (IID exec-accept ~84% vs random ~2%).

## Hard L1 + template sampling

Freeze defaults (k=6, stmts 5–9, deps≥3, ordered `add∘mul`): `docs/l1-hard-grammar.md`  
Template front-end + two-arm N curve: `docs/template-sampling.md`

```bash
# Two-arm N(k=6) curve (hard vs mixture)
PYTHONPATH=. python3 -m src.experiments.n_curve \
  --arms hard mixture --n-grid 200 500 1000 --epochs 10 \
  --out results/n_curve_k6

# Single run
PYTHONPATH=. python3 -m src.train --arm hard --n-train 1000 --epochs 15 \
  --block-size 128 --out results/runs/hard_n1000
```

Historical difficulty-gate (~60% with raised knobs): `results/difficulty_gate_k6_n1000.json` — knobs ≠ freeze.

**Next (local 4×GPU / WSL):** longer N grid + multi-seed.

## Layout

```text
src/dsl/       # tokens, L0/L1 interpreters, legal continuations
src/data/      # samplers + IID / hold-out (toy hold-out today)
src/model/     # MiniGPT (~4×128)
src/train.py   # NTP training
src/eval.py    # legal-next, exec-accept (sampled; full free-gen jsonl)
docs/          # plan pointer + migration note
tests/
results/       # sanity archive; grid runs go under grid_kN/
                 # each run also has train.jsonl + free_gen_*.jsonl
```

## Metrics

| Metric | Meaning |
|--------|---------|
| `legal_next_rate` | P(argmax next ∈ interpreter legal set) |
| `exec_accept_rate` | Sampled autoregressive program accepted by interpreter (**main**) |
| hold-out split | Same on A∘B programs never seen in training |

“Learned”: IID `exec_accept_rate` ≥ 90%. Critical N = smallest N that crosses that line at fixed k.

## Run dumps (full free-gen + train)

`src.train` (including every `job.sh` grid/smoke cell) writes a self-contained `--out` folder. `history.json` still stores only a **5-program preview** in `metrics.<split>.examples`. The full sample is on disk.

| File | Contents |
|------|----------|
| `train.jsonl` | Copy of the full training split (`data/train.jsonl`, or the file from `--data-dir`). |
| `free_gen_test_iid.jsonl` | Every free-gen program from the **latest** exec-accept eval on IID. |
| `free_gen_test_holdout.jsonl` | Same for the hold-out split, when that split is non-empty. |
| `free_gen_<split>_epoch<E>.jsonl` | Full dump for eval at epoch E (not overwritten by later evals). |
| `history.json` | Per-epoch metrics. `examples` is the 5-row preview; `examples_path` points at the epoch jsonl. |
| `run_meta.json` | `train_jsonl`, `train_jsonl_source`, `n_train_saved`, `data_dir`, and `free_gen` (latest path + line count). |

Each free-gen jsonl row has `step`, `tokens` (including `<bos>` / `<eos>`), `ok`, `error`, `n_new_tokens`, plus `seed`, `epoch`, and `split` when the run supplies them.

`exec_accept_rate`, `n_unique_programs`, and legal-next rates are still computed on the full sample, not the preview. Pass `--no-dump-free-gen` to skip the jsonl dumps; the train copy is still written. `job.sh` passes `--dump-free-gen` explicitly. Standalone eval: `python3 -m src.eval --checkpoint ... --data-dir ... --out <run_dir>`.
