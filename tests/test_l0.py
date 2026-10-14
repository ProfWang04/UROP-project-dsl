"""L0 arithmetic interpreter + generator tests."""

from __future__ import annotations

import random

from src.dsl.l0 import evaluate_l0, generate_l0, legal_l0_continuations
from src.dsl.tokens import EOS


def test_l0_simple_eval():
    assert evaluate_l0(["2", "+", "3"]).ok
    assert evaluate_l0(["2", "+", "3"]).value == 5
    assert evaluate_l0(["(", "2", "+", "3", ")", "*", "4"]).value == 20
    assert evaluate_l0(["9", "-", "1", "*", "2"]).value == 7  # 9-(1*2)


def test_l0_negative_examples():
    assert not evaluate_l0(["+", "1"]).ok
    assert not evaluate_l0(["1", "+"]).ok
    assert not evaluate_l0(["(", "1"]).ok
    assert not evaluate_l0(["1", ")"]).ok
    assert not evaluate_l0([]).ok


def test_l0_generator_always_valid():
    rng = random.Random(0)
    for _ in range(30):
        toks = generate_l0(rng, max_depth=3)
        result = evaluate_l0(toks)
        assert result.ok, result.error


def test_l0_legal_continuations_start():
    cont = legal_l0_continuations([])
    assert "1" in cont and "(" in cont
    assert "+" not in cont


def test_l0_legal_after_digit():
    cont = legal_l0_continuations(["1"])
    assert "+" in cont and EOS in cont
    assert "2" not in cont
