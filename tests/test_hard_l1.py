"""Hard L1 freeze: deps + strict ordered hold-out."""

from __future__ import annotations

import random

from src.dsl.hard_l1 import (
    HARD_API_K,
    HARD_HOLDOUT_PAIR,
    FREEZE_MIN_DEPS,
    hard_l1_config,
    generate_hard_l1,
    generate_hard_holdout_combo,
    has_ordered_api_pair,
    count_cross_stmt_deps,
)
from src.dsl.l1 import L1Interpreter, evaluate_l1
from src.data.generate import generate_l1_dataset


def test_hard_apis_k6():
    cfg = hard_l1_config()
    assert cfg.api_k == HARD_API_K == 6
    assert cfg.apis == ["set", "print", "inc", "dec", "add", "mul"]
    assert HARD_HOLDOUT_PAIR == ("add", "mul")
    assert cfg.min_stmts == 5 and cfg.max_stmts == 9
    assert FREEZE_MIN_DEPS == 3


def test_ordered_pair_detection():
    assert has_ordered_api_pair(["add", "a", "b", "mul", "a", "b"], "add", "mul")
    assert not has_ordered_api_pair(["mul", "a", "b", "add", "a", "b"], "add", "mul")
    assert not has_ordered_api_pair(["add", "a", "b"], "add", "mul")


def test_hard_generator_valid_with_deps():
    rng = random.Random(0)
    cfg = hard_l1_config()
    for _ in range(30):
        toks = generate_hard_l1(cfg, rng, min_deps=FREEZE_MIN_DEPS)
        assert L1Interpreter(toks, apis=cfg.apis).run().ok
        assert count_cross_stmt_deps(toks, cfg.apis) >= FREEZE_MIN_DEPS


def test_hard_holdout_combo_contains_both():
    rng = random.Random(1)
    cfg = hard_l1_config()
    a, b = HARD_HOLDOUT_PAIR
    for _ in range(20):
        toks = generate_hard_holdout_combo(cfg, a, b, rng)
        assert has_ordered_api_pair(toks, a, b)
        assert evaluate_l1(toks, api_k=6).ok


def test_hard_dataset_split_invariants():
    split = generate_l1_dataset(
        n_train=80, n_test_iid=20, n_test_holdout=20, seed=0, hard=True, arm="hard"
    )
    a, b = split.meta["holdout_pair"]
    assert split.meta["holdout_mode"] == "ordered_A_then_B"
    assert split.meta["api_k"] == 6
    assert split.meta["freeze_defaults"]["min_deps"] == 3

    for seq in split.train + split.test_iid:
        body = [t for t in seq if t not in ("<bos>", "<eos>")]
        assert not has_ordered_api_pair(body, a, b)
        assert evaluate_l1(body, api_k=6).ok

    for seq in split.test_holdout:
        body = [t for t in seq if t not in ("<bos>", "<eos>")]
        assert has_ordered_api_pair(body, a, b)
        assert evaluate_l1(body, api_k=6).ok
