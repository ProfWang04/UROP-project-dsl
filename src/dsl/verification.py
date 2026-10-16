"""Versioned framing, completion-prefix, and split acceptance rules.

Kept independent of PyTorch so the experimental contract can be tested alone.
"""

from __future__ import annotations

from .tokens import BOS, EOS, PAD, UNK, enabled_apis
from .l0 import evaluate_l0
from .l1 import evaluate_l1
from .hard_l1 import accept_hard_program

EVALUATION_VERSION = 2


def completion_prefix(seq: list[str], level: str, api_k: int) -> list[str]:
    """L1: first complete statement, excluding the next opcode; L0: BOS."""
    if level == "l0":
        return [BOS]
    apis = set(enabled_apis(api_k))
    for i in range(2, len(seq) - 1):
        if seq[i] in apis:
            return seq[:i]
    # A one-statement program supplies its opcode only, never its whole body.
    return seq[:2]


def check_generated_program(
    tokens: list[str], level: str, api_k: int,
    dataset_meta: dict | None = None, split: str | None = None,
) -> tuple[bool, bool, str | None]:
    """Return (split-task success, framed interpreter validity, failure reason)."""
    if not tokens or tokens[0] != BOS or tokens[-1] != EOS:
        return False, False, "missing_eos_or_bos"
    body = tokens[1:-1]
    if any(t in (BOS, EOS, PAD, UNK) for t in body):
        return False, False, "special_token_in_body"
    result = evaluate_l0(body) if level == "l0" else evaluate_l1(body, api_k=api_k)
    if not result.ok:
        return False, False, result.error
    meta = dataset_meta or {}
    if level == "l1" and split in ("test_iid", "test_holdout"):
        pair = meta.get("holdout_pair")
        if meta.get("hard"):
            knobs = meta.get("experiment_knobs", {})
            ok, reason = accept_hard_program(
                body, apis=enabled_apis(api_k), api_k=api_k,
                min_deps=meta.get("min_deps", knobs.get("min_deps", 3)),
                min_distinct_apis=knobs.get("min_distinct_apis", 0),
                require_ordered_holdout=split == "test_holdout",
                forbid_ordered_holdout=split == "test_iid",
                holdout_pair=tuple(pair or ("add", "mul")),
            )
            if not ok:
                return False, True, reason
            minimum = knobs.get("min_stmts", 5)
            if sum(t in enabled_apis(api_k) for t in body) < minimum:
                return False, True, "too_few_statements"
        elif pair:
            both = all(api in body for api in pair)
            if both != (split == "test_holdout"):
                return False, True, "holdout_membership"
    return True, True, None


