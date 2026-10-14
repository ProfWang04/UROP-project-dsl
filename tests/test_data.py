"""Dataset generation and hold-out invariants."""

from __future__ import annotations

from src.data.generate import generate_l1_dataset, generate_l0_dataset
from src.dsl.l1 import evaluate_l1
from src.dsl.l0 import evaluate_l0
from src.dsl.tokens import BOS, EOS


def test_l0_dataset_valid():
    split = generate_l0_dataset(n_train=50, n_test=20, seed=0)
    assert len(split.train) == 50
    for seq in split.train + split.test_iid:
        body = [t for t in seq if t not in (BOS, EOS)]
        assert evaluate_l0(body).ok


def test_l1_holdout_disjoint_combo():
    split = generate_l1_dataset(n_train=100, api_k=4, n_test_iid=30, n_test_holdout=30, seed=0)
    a, b = split.meta["holdout_pair"]

    def both(seq):
        return a in seq and b in seq

    for seq in split.train + split.test_iid:
        assert not both(seq)
        body = [t for t in seq if t not in (BOS, EOS)]
        assert evaluate_l1(body, api_k=4).ok

    for seq in split.test_holdout:
        assert both(seq)
        body = [t for t in seq if t not in (BOS, EOS)]
        assert evaluate_l1(body, api_k=4).ok
