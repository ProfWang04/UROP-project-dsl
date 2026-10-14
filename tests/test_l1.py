"""L1 interpreter, generator, and legal continuations."""

from __future__ import annotations

import random

from src.dsl.l1 import L1Config, L1Interpreter, evaluate_l1, generate_l1, generate_l1_holdout_combo
from src.dsl.continuations import legal_continuations, is_prefix_legal
from src.dsl.tokens import EOS, enabled_apis


def test_l1_set_print():
    r = evaluate_l1(["set", "a", "3", "print", "a"], api_k=3)
    assert r.ok
    assert r.outputs == [3]
    assert r.env["a"] == 3


def test_l1_expr_with_var():
    r = evaluate_l1(["set", "a", "2", "set", "b", "a", "+", "3", "print", "b"], api_k=3)
    assert r.ok
    assert r.outputs == [5]


def test_l1_extra_apis():
    r = evaluate_l1(["set", "a", "1", "inc", "a", "print", "a"], api_k=3)
    assert r.ok
    assert r.outputs == [2]


def test_l1_negative_unknown_api():
    # clr requires api_k >= 6 (set,print,inc,dec,add,mul,clr) — with k=3, clr disabled
    r = evaluate_l1(["clr", "a"], api_k=3)
    assert not r.ok


def test_l1_negative_syntax():
    assert not evaluate_l1(["set"], api_k=3).ok
    assert not evaluate_l1(["set", "a"], api_k=3).ok
    assert not evaluate_l1(["print"], api_k=3).ok
    assert not evaluate_l1(["inc"], api_k=3).ok


def test_l1_generator_valid():
    rng = random.Random(1)
    cfg = L1Config(api_k=3, max_stmts=4)
    for _ in range(40):
        toks = generate_l1(cfg, rng)
        r = L1Interpreter(toks, apis=cfg.apis).run()
        assert r.ok, (toks, r.error)


def test_l1_holdout_combo_contains_both():
    rng = random.Random(2)
    cfg = L1Config(api_k=4)
    a, b = cfg.apis[2], cfg.apis[1]
    for _ in range(20):
        toks = generate_l1_holdout_combo(cfg, a, b, rng)
        assert a in toks and b in toks
        assert L1Interpreter(toks, apis=cfg.apis).run().ok


def test_l1_legal_continuations():
    assert "set" in legal_continuations([], level="l1", api_k=3)
    cont = legal_continuations(["set"], level="l1", api_k=3)
    assert "a" in cont
    cont = legal_continuations(["set", "a"], level="l1", api_k=3)
    assert "1" in cont or "(" in cont
    cont = legal_continuations(["set", "a", "1"], level="l1", api_k=3)
    assert EOS in cont
    assert "print" in cont


def test_l1_prefix_legal():
    assert is_prefix_legal(["set", "a", "1"], level="l1", api_k=3)
    assert is_prefix_legal(["set", "a", "1", EOS], level="l1", api_k=3)
    assert not is_prefix_legal(["set", "+"], level="l1", api_k=3)
    assert not is_prefix_legal(["1", "set"], level="l1", api_k=3)


def test_enabled_apis_k():
    assert enabled_apis(2) == ["set", "print"]
    assert enabled_apis(3) == ["set", "print", "inc"]
    assert "mov" in enabled_apis(8)
