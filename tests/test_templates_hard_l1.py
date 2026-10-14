"""Cadmus-style template front-end tests."""

from __future__ import annotations

import random

from src.dsl.hard_l1 import (
    HARD_API_K,
    HARD_HOLDOUT_PAIR,
    FREEZE_MIN_DEPS,
    FREEZE_MIN_STMTS,
    FREEZE_MAX_STMTS,
    hard_l1_config,
    has_ordered_api_pair,
    accept_hard_program,
)
from src.dsl.templates_hard_l1 import (
    TEMPLATES,
    instantiate,
    sample_with_templates,
    sample_mixture_curriculum,
    SampleStats,
)
from src.data.generate import generate_l1_dataset, holdout_pair_hash


def test_freeze_defaults_not_gate_knobs():
    cfg = hard_l1_config()
    assert cfg.api_k == HARD_API_K == 6
    assert cfg.min_stmts == FREEZE_MIN_STMTS == 5
    assert cfg.max_stmts == FREEZE_MAX_STMTS == 9
    assert FREEZE_MIN_DEPS == 3
    assert HARD_HOLDOUT_PAIR == ("add", "mul")


def test_all_templates_instantiate_and_often_accept():
    rng = random.Random(0)
    cfg = hard_l1_config()
    ok_count = 0
    for spec in TEMPLATES:
        for _ in range(5):
            toks = instantiate(spec.template_id, rng, cfg.vars)
            ok, _ = accept_hard_program(
                toks,
                apis=cfg.apis,
                min_deps=1,
                require_ordered_holdout=True if spec.family == "holdout-combo" else None,
                forbid_ordered_holdout=False,
            )
            if ok:
                ok_count += 1
                break
    assert ok_count >= 6  # most families should have a valid fill


def test_sample_with_templates_train_forbids_ordered():
    rng = random.Random(1)
    stats = SampleStats()
    for _ in range(20):
        toks, tid = sample_with_templates(
            rng, holdout=False, min_deps=3, stats=stats
        )
        assert not has_ordered_api_pair(toks, "add", "mul"), (toks, tid)
    assert stats.accepts >= 20


def test_sample_with_templates_holdout_requires_ordered():
    rng = random.Random(2)
    for _ in range(15):
        toks, _tid = sample_with_templates(rng, holdout=True, min_deps=3)
        assert has_ordered_api_pair(toks, "add", "mul")


def test_mixture_returns_k():
    rng = random.Random(3)
    ks = set()
    for _ in range(30):
        _toks, _tid, k = sample_mixture_curriculum(rng)
        ks.add(k)
    assert ks & {2, 3, 6}


def test_hard_dataset_manifest_fields():
    split = generate_l1_dataset(
        n_train=40, n_test_iid=10, n_test_holdout=10, seed=0, hard=True, arm="hard"
    )
    m = split.meta
    assert m["freeze_defaults"]["min_deps"] == 3
    assert m["freeze_defaults"]["min_stmts"] == 5
    assert "gate_knobs_reference" in m
    assert m["holdout_pair_hash"] == holdout_pair_hash(("add", "mul"))
    assert m["sample_stats"]["train"]["proposals"] >= m["sample_stats"]["train"]["accepts"]
    a, b = m["holdout_pair"]
    for seq in split.train + split.test_iid:
        body = [t for t in seq if t not in ("<bos>", "<eos>")]
        assert not has_ordered_api_pair(body, a, b)
    for seq in split.test_holdout:
        body = [t for t in seq if t not in ("<bos>", "<eos>")]
        assert has_ordered_api_pair(body, a, b)
