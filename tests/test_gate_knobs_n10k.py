"""Gate-knobs N=10k hang fix: uniqueness, progress, hold-out invariants."""

from __future__ import annotations

import time

from src.data.generate import generate_l1_dataset
from src.dsl.hard_l1 import has_ordered_api_pair, HARD_HOLDOUT_PAIR, GATE_KNOBS
from src.dsl.l1 import evaluate_l1


def _bodies(split_seqs):
    return [[t for t in s if t not in ("<bos>", "<eos>")] for s in split_seqs]


def test_gate_knobs_n1000_finishes_unique_and_holdout():
    t0 = time.time()
    split = generate_l1_dataset(
        n_train=1000,
        n_test_iid=100,
        n_test_holdout=100,
        seed=0,
        hard=True,
        arm="hard",
        gate_knobs=True,
    )
    elapsed = time.time() - t0
    assert elapsed < 120, elapsed
    assert len(split.train) == 1000
    assert len({tuple(s) for s in split.train}) == 1000
    assert split.meta["experiment_knobs"]["min_deps"] == GATE_KNOBS["min_deps"]
    # freeze defaults unchanged in meta
    assert split.meta["freeze_defaults"]["min_deps"] == 3
    assert split.meta["freeze_defaults"]["min_stmts"] == 5

    a, b = HARD_HOLDOUT_PAIR
    for body in _bodies(split.train + split.test_iid):
        assert evaluate_l1(body, api_k=6).ok
        assert not has_ordered_api_pair(body, a, b)
    for body in _bodies(split.test_holdout):
        assert evaluate_l1(body, api_k=6).ok
        assert has_ordered_api_pair(body, a, b)

    accepts = split.meta["sample_stats"]["train"]["template_accepts"]
    total = sum(accepts.values()) or 1
    # Not stuck on a single short template family; procedural OK as majority
    # but mix should include ≥1 non-procedural OR ≥2 procedural tags.
    non_proc = sum(v for k, v in accepts.items() if not k.startswith("procedural"))
    assert non_proc >= 50 or len(accepts) >= 2, accepts
    reuse_only = accepts.get("reuse.same_var_loopish", 0) / total
    assert reuse_only < 0.9, accepts


def test_gate_knobs_n10000_finishes():
    t0 = time.time()
    split = generate_l1_dataset(
        n_train=10000,
        n_test_iid=200,
        n_test_holdout=200,
        seed=1,
        hard=True,
        arm="hard",
        gate_knobs=True,
    )
    elapsed = time.time() - t0
    assert elapsed < 180, elapsed
    assert len(split.train) == 10000
    assert len({tuple(s) for s in split.train}) == 10000
    a, b = HARD_HOLDOUT_PAIR
    # spot-check hold-out invariants
    for body in _bodies(split.train[:50]):
        assert not has_ordered_api_pair(body, a, b)
        assert evaluate_l1(body, api_k=6).ok
    for body in _bodies(split.test_holdout[:50]):
        assert has_ordered_api_pair(body, a, b)
        assert evaluate_l1(body, api_k=6).ok
    accepts = split.meta["sample_stats"]["train"]["template_accepts"]
    non_proc = sum(v for k, v in accepts.items() if not str(k).startswith("procedural"))
    assert non_proc >= 100 or len(accepts) >= 2, accepts
