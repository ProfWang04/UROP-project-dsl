# Hard L1 grammar freeze

**Status:** freeze defaults locked; Cadmus template front-end on top.  
Prior difficulty-gate (~60% with raised stmts/deps) = **experimental knobs**, not freeze.

## Freeze defaults (code constants)

| Knob | Value |
|------|--------|
| `HARD_API_K` | 6 → `set print inc dec add mul` |
| `HARD_HOLDOUT_PAIR` | ordered `add ∘ mul` |
| n_vars | 6 (`a`–`f`) |
| min/max stmts | **5 / 9** |
| expr_depth | 3 |
| min_deps | **3** |

EXTRA order: `inc, dec, add, mul, clr, mov` (+ optional later `neg, swap` for k≥9 — not required for freeze).

## Gate knobs (manifest only)

Historical difficulty gate used stmts 10–16, min_deps 7, min_distinct_apis 4. See `GATE_KNOBS` / `--gate-knobs`.

## Sampling

Template front-end: `docs/template-sampling.md`. Acceptance unchanged (interpreter + deps + ordered hold-out).
