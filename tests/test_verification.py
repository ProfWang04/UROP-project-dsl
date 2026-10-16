"""The scoring contract must distinguish validity from split-task success."""

import pytest

from src.data.generate import generate_dataset, main as generate_main
from src.dsl.tokens import BOS, EOS, PAD, UNK
from src.dsl.verification import completion_prefix, check_generated_program


def wrap(body):
    return [BOS, *body.split(), EOS]


@pytest.mark.parametrize("tokens", [
    [BOS, "1"], [BOS, "1", BOS, EOS],
    [BOS, "1", PAD, EOS], [BOS, "1", UNK, EOS],
    [BOS, "1", EOS, EOS],
])
def test_invalid_framing_is_rejected(tokens):
    assert check_generated_program(tokens, "l0", 3)[:2] == (False, False)


def test_holdout_requires_composition_even_if_interpreter_accepts():
    meta = {"hard": True, "holdout_pair": ["add", "mul"], "min_deps": 3,
            "experiment_knobs": {"min_stmts": 5}}
    iid = wrap("set a 1 inc a inc a print a inc a")
    holdout = wrap("set a 1 inc a add a a mul a a print a")
    assert check_generated_program(iid, "l1", 6, meta, "test_iid")[:2] == (True, True)
    assert check_generated_program(iid, "l1", 6, meta, "test_holdout")[:2] == (False, True)
    assert check_generated_program(holdout, "l1", 6, meta, "test_holdout")[:2] == (True, True)
    assert check_generated_program(holdout, "l1", 6, meta, "test_iid")[:2] == (False, True)
    assert check_generated_program(wrap("print 1"), "l1", 6, meta, "test_iid")[:2] == (False, True)


def test_soft_holdout_requires_both_apis():
    meta = {"hard": False, "holdout_pair": ["print", "inc"]}
    seq = wrap("set a 1 print a")
    assert check_generated_program(seq, "l1", 3, meta, "test_iid")[0]
    assert not check_generated_program(seq, "l1", 3, meta, "test_holdout")[0]


def test_hard_minimum_statement_count_and_api_diversity():
    meta = {"hard": True, "min_deps": 3, "experiment_knobs": {"min_stmts": 5}}
    short = wrap("set a 1 inc a inc a inc a")
    assert check_generated_program(short, "l1", 6, meta, "test_iid") == (
        False, True, "too_few_statements")
    meta["experiment_knobs"]["min_distinct_apis"] = 4
    narrow = wrap("set a 1 inc a inc a inc a inc a")
    assert check_generated_program(narrow, "l1", 6, meta, "test_iid") == (
        False, True, "distinct_apis")


def test_prefix_uses_input_without_leaking_reference_suffix():
    assert completion_prefix(wrap("set a ( 1 + 2 ) add a a mul a a"), "l1", 6) == [BOS, "set", "a", "(", "1", "+", "2", ")"]
    assert completion_prefix(wrap("set b 9 print b"), "l1", 6) == [BOS, "set", "b", "9"]
    assert completion_prefix(wrap("print 1"), "l1", 3) == [BOS, "print"]
    assert completion_prefix(wrap("1 + 2"), "l0", 3) == [BOS]


def test_l0_dispatch_accepts_shared_cli_options(tmp_path):
    split = generate_dataset(level="l0", n_train=2, n_test=2,
                             gate_knobs=False, use_templates=True)
    assert len(split.train) == 2
    generate_main(["--level", "l0", "--n-train", "2", "--out", str(tmp_path)])
    assert (tmp_path / "meta.json").is_file()
