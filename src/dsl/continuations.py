"""Legal next-token sets from the interpreter (stepwise verifier)."""

from __future__ import annotations

from .l0 import legal_l0_continuations, evaluate_l0
from .l1 import L1Interpreter, L1Config, evaluate_l1
from .tokens import (
    BOS,
    EOS,
    DIGITS,
    OPS,
    VARS,
    enabled_apis,
)


def legal_continuations(
    prefix: list[str],
    level: str = "l1",
    api_k: int = 3,
) -> set[str]:
    """
    Return the set of tokens that can legally follow `prefix`.

    For L1 we use a structural state machine over statements + L0-ish exprs.
    Empty set means the prefix is already illegal (or complete with no EOS — rare).
    """
    clean = [t for t in prefix if t != BOS]
    if clean and clean[-1] == EOS:
        return set()

    if level == "l0":
        return legal_l0_continuations(clean)

    return _legal_l1(clean, api_k)


def _legal_l1(prefix: list[str], api_k: int) -> set[str]:
    apis = set(enabled_apis(api_k))
    vars_set = set(VARS)

    if not prefix:
        return set(apis)

    # Walk statements; if we finish all complete stmts, next is API or EOS.
    i = 0
    n = len(prefix)
    completed_any = False

    while i < n:
        op = prefix[i]
        if op not in apis:
            return set()  # illegal
        i += 1
        need = _stmt_needs(op)
        # Consume required fixed args (vars)
        for kind in need["fixed"]:
            if i >= n:
                # Still expecting this kind
                return _expect_set(kind, vars_set, apis)
            tok = prefix[i]
            if kind == "var" and tok not in vars_set:
                return set()
            i += 1

        if need["expr"]:
            # Parse expression from i; may be incomplete
            expr_start = i
            cont, consumed, complete = _expr_step(prefix, i, apis)
            if consumed < 0:
                return set()  # illegal expr
            i = expr_start + consumed
            if not complete:
                # Mid-expression: return expression continuations
                # Also if expression can end and we're at end of prefix, cont is next expr tokens
                return cont
        completed_any = True

    # All complete statements consumed; next is new stmt or EOS
    if completed_any:
        return set(apis) | {EOS}
    return set(apis)


def _stmt_needs(op: str) -> dict:
    if op == "set":
        return {"fixed": ["var"], "expr": True}
    if op == "print":
        return {"fixed": [], "expr": True}
    if op in ("inc", "dec", "clr", "neg"):
        return {"fixed": ["var"], "expr": False}
    if op in ("add", "mul", "mov", "swap"):
        return {"fixed": ["var", "var"], "expr": False}
    return {"fixed": [], "expr": False}


def _expect_set(kind: str, vars_set: set[str], apis: set[str]) -> set[str]:
    if kind == "var":
        return set(vars_set)
    return set()


def _expr_step(
    tokens: list[str], start: int, apis: set[str]
) -> tuple[set[str], int, bool]:
    """
    Walk expression starting at `start`.

    Returns (continuations_if_incomplete, tokens_consumed, is_complete).
    consumed = -1 on illegal.
    """
    i = start
    n = len(tokens)
    balance = 0
    # States after each token: expect_atom | expect_op_or_end
    expect_atom = True

    while i < n:
        tok = tokens[i]
        # Boundary: next statement API ends expression (only if we have a value and balance 0)
        if tok in apis and balance == 0 and not expect_atom:
            return set(), i - start, True

        if expect_atom:
            if tok in DIGITS or tok in VARS:
                expect_atom = False
                i += 1
                continue
            if tok == "(":
                balance += 1
                i += 1
                continue
            return set(), -1, False
        else:
            if tok in OPS:
                expect_atom = True
                i += 1
                continue
            if tok == ")" and balance > 0:
                balance -= 1
                i += 1
                continue
            if tok in apis and balance == 0:
                return set(), i - start, True
            return set(), -1, False

    # Incomplete at end of prefix
    if expect_atom:
        cont: set[str] = set(DIGITS) | set(VARS) | {"("}
        return cont, i - start, False

    cont = set(OPS)
    if balance > 0:
        cont.add(")")
    if balance == 0:
        # Expression can end → next stmt API or EOS
        cont |= set(apis) | {EOS}
    return cont, i - start, False


def is_prefix_legal(prefix: list[str], level: str = "l1", api_k: int = 3) -> bool:
    """True if some continuation (possibly EOS) can complete a valid program."""
    clean = [t for t in prefix if t != BOS]
    if not clean:
        return True
    if clean[-1] == EOS:
        body = clean[:-1]
        if level == "l0":
            return evaluate_l0(body).ok
        return evaluate_l1(body, api_k=api_k).ok
    # Has any legal next token?
    return len(legal_continuations(clean, level=level, api_k=api_k)) > 0
