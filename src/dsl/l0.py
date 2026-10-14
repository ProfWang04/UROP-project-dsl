"""L0: arithmetic expressions with + - * ( ) and single-digit integers."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .tokens import DIGITS, OPS, PARENS, EOS, BOS


@dataclass
class L0Result:
    ok: bool
    value: int | None = None
    error: str | None = None
    tokens: list[str] | None = None


class L0Interpreter:
    """Recursive-descent parser/evaluator for L0 expressions."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> str | None:
        if self.i >= len(self.tokens):
            return None
        return self.tokens[self.i]

    def consume(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("unexpected end of expression")
        if expected is not None and tok != expected:
            raise SyntaxError(f"expected {expected!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse_expr(self) -> int:
        # expr := term (("+" | "-") term)*
        val = self.parse_term()
        while self.peek() in ("+", "-"):
            op = self.consume()
            rhs = self.parse_term()
            val = val + rhs if op == "+" else val - rhs
        return val

    def parse_term(self) -> int:
        # term := factor ("*" factor)*
        val = self.parse_factor()
        while self.peek() == "*":
            self.consume("*")
            val = val * self.parse_factor()
        return val

    def parse_factor(self) -> int:
        # factor := DIGIT | "(" expr ")"
        tok = self.peek()
        if tok is None:
            raise SyntaxError("unexpected end in factor")
        if tok in DIGITS:
            self.consume()
            return int(tok)
        if tok == "(":
            self.consume("(")
            val = self.parse_expr()
            self.consume(")")
            return val
        raise SyntaxError(f"unexpected token in factor: {tok!r}")

    def run(self) -> L0Result:
        try:
            value = self.parse_expr()
            if self.i != len(self.tokens):
                return L0Result(
                    ok=False,
                    error=f"trailing tokens from index {self.i}: {self.tokens[self.i:]}",
                    tokens=self.tokens,
                )
            return L0Result(ok=True, value=value, tokens=self.tokens)
        except (SyntaxError, ValueError) as e:
            return L0Result(ok=False, error=str(e), tokens=self.tokens)


def evaluate_l0(tokens: list[str]) -> L0Result:
    """Evaluate a token list as an L0 expression (no BOS/EOS)."""
    clean = [t for t in tokens if t not in (BOS, EOS)]
    return L0Interpreter(clean).run()


def generate_l0(
    rng: random.Random,
    max_depth: int = 3,
    ops: list[str] | None = None,
) -> list[str]:
    """Sample a guaranteed-valid L0 expression as tokens."""
    ops = ops or list(OPS)

    def gen(depth: int) -> list[str]:
        if depth <= 0 or rng.random() < 0.35:
            return [rng.choice(DIGITS)]
        if rng.random() < 0.25:
            # parenthesized subexpr
            inner = gen(depth - 1)
            return ["("] + inner + [")"]
        # binary op
        left = gen(depth - 1)
        right = gen(depth - 1)
        op = rng.choice(ops)
        # sometimes wrap whole thing
        tokens = left + [op] + right
        if rng.random() < 0.2 and depth > 1:
            tokens = ["("] + tokens + [")"]
        return tokens

    return gen(max_depth)


def legal_l0_continuations(prefix: list[str]) -> set[str]:
    """
    Approximate legal next tokens for an L0 expression prefix.
    Conservative: tokens that keep some valid completion possible.
    """
    clean = [t for t in prefix if t not in (BOS, EOS)]
    if not clean:
        return set(DIGITS) | {"("}

    # Track paren balance and last kind
    balance = 0
    for t in clean:
        if t == "(":
            balance += 1
        elif t == ")":
            balance -= 1
        if balance < 0:
            return set()  # already illegal

    last = clean[-1]
    digit_set = set(DIGITS)
    can_end = balance == 0 and last in digit_set | {")"}

    if last in digit_set or last == ")":
        # after value: op, or close paren, or EOS if balanced
        cont: set[str] = set(OPS)
        if balance > 0:
            cont.add(")")
        if can_end:
            cont.add(EOS)
        return cont

    if last in OPS or last == "(":
        return digit_set | {"("}

    return set()
