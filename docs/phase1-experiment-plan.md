# Phase 1 experiment plan

**Canonical (store):** [`/cursor/stores/self/docs/phase1-experiment-plan.md`](/cursor/stores/self/docs/phase1-experiment-plan.md)

## Status of this repo

Current code is the **toy / pipeline slice** (L0 + easy L1, k=3, N=1e3 sanity ≫ random). That run is a **historical positive baseline**, not the main Phase 1 claim.

**Main Phase 1 direction now:** harder L1 (`k ∈ {4,6,8,10}`, strict composition hold-out, longer variable dependencies) × full `N ∈ {1e3,1e4,1e5}` × ≥3 seeds on ~4×GPU — critical learnability curves (exec-accept).

## Migration (keep vs replace)

| Keep | Upgrade next |
|------|----------------|
| Interpreter-as-verifier metrics, MiniGPT NTP skeleton | Hard L1 generator + strict A∘B hold-out |
| `results` sanity artifacts (do not overwrite) | `results/grid_kN/` multi-seed jobs |
| L0 / unit tests as CI smoke | Early-stopping + grid scripts |

Do not treat high exec-accept on k=3 N=1e3 as “Phase 1 done.”
