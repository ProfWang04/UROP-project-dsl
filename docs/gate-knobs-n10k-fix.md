# Gate-knobs N=10k hang fix

Canonical store copy: `/cursor/stores/self/docs/gate-knobs-n10k-fix.md`

Under `--gate-knobs`, short templates collapsed to ~3k uniques → infinite dedupe loop. Fixed with gate-length templates, pad-toward-knobs, hold-out-aware salted procedural, `force_unique` escape, and progress logs. Freeze defaults unchanged.

```bash
PYTHONPATH=. python3 -m src.data.generate --hard --gate-knobs --n-train 10000 --out /tmp/gk_n10k
PYTHONPATH=. python3 -m pytest tests/test_gate_knobs_n10k.py -q
```
