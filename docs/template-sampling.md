# Cadmus-style template sampling (hard L1 front-end)

Templates **propose** candidates; **acceptance** is unchanged:

1. `evaluate_l1(...).ok`
2. cross-stmt deps ≥ freeze `min_deps` (default **3**)
3. train/IID: **forbid** ordered `add∘mul`
4. hold-out pool: **require** ordered `add∘mul`

## Freeze defaults vs gate knobs

| | Freeze (code constants) | Gate knobs (experimental / historical) |
|--|-------------------------|----------------------------------------|
| k | 6 | — |
| hold-out | ordered `("add","mul")` | — |
| stmts | 5–9 | 10–16 |
| min_deps | 3 | 7 |
| min_distinct_apis | 0 (off) | 4 |

Gate knobs live in `GATE_KNOBS` / `--gate-knobs` and in manifests as `gate_knobs_reference` — they do **not** change freeze defaults.

## Template families

| Family | IDs |
|--------|-----|
| pipeline | `pipe.chain_set_add_inc_print`, `pipe.dec_mul_print` |
| fan-in | `fanin.two_src_add`, `fanin.three_src_mul` |
| reuse | `reuse.same_var_loopish`, `reuse.dual_pingpong` |
| holdout-combo | `holdout.add_then_mul`, `holdout.add_filler_mul` |

Code: `src/dsl/templates_hard_l1.py`.

## Two arms

1. **hard** — train/IID/hold-out all from k=6 template+gates  
2. **mixture** — train mixes k'∈{2,3,6}; IID+hold-out eval still hard k=6 ordered `add∘mul`

## Run N(k=6) curve

```bash
# Small CPU sweep (example)
PYTHONPATH=. python3 -m src.experiments.n_curve \
  --arms hard mixture \
  --n-grid 200 500 1000 \
  --epochs 10 \
  --max-eval 40 \
  --out results/n_curve_k6

# Single arm / single N via train
PYTHONPATH=. python3 -m src.train --hard --arm hard --n-train 1000 \
  --epochs 15 --block-size 128 --out results/runs/hard_n1000
PYTHONPATH=. python3 -m src.train --arm mixture --n-train 1000 \
  --epochs 15 --block-size 128 --out results/runs/mix_n1000
```

Outputs per run: `data/manifest.json` (template IDs, reject rates, `holdout_pair_hash`, freeze vs knobs).  
Curve rollup: `results/n_curve_k6/curve_manifest.json` with `N_star_by_arm`.

Primary metrics: **exec-accept** (IID + hold-out). Secondary: legal-next.
