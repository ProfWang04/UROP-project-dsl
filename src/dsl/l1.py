"""L1: imperative command DSL — set / print / optional extra APIs."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .l0 import L0Interpreter, generate_l0
from .tokens import (
    BOS,
    EOS,
    VARS,
    DIGITS,
    CORE_APIS,
    EXTRA_APIS,
    enabled_apis,
)


@dataclass
class L1Config:
    """Generation / interpreter config for L1."""

    api_k: int = 3
    n_vars: int = 4
    max_stmts: int = 5
    min_stmts: int = 1
    expr_depth: int = 2
    seed: int | None = None

    @property
    def apis(self) -> list[str]:
        return enabled_apis(self.api_k)

    @property
    def vars(self) -> list[str]:
        return VARS[: max(1, min(self.n_vars, len(VARS)))]


@dataclass
class L1Result:
    ok: bool
    env: dict[str, int] = field(default_factory=dict)
    outputs: list[int] = field(default_factory=list)
    error: str | None = None
    tokens: list[str] | None = None


def _eval_expr_with_env(tokens: list[str], env: dict[str, int]) -> int:
    """L0-style eval where VAR tokens are atoms looking up env."""

    class EnvInterp(L0Interpreter):
        def parse_factor(self) -> int:  # type: ignore[override]
            tok = self.peek()
            if tok is None:
                raise SyntaxError("unexpected end in factor")
            if tok in DIGITS:
                self.consume()
                return int(tok)
            if tok in VARS:
                self.consume()
                return int(env.get(tok, 0))
            if tok == "(":
                self.consume("(")
                val = self.parse_expr()
                self.consume(")")
                return val
            raise SyntaxError(f"unexpected token in factor: {tok!r}")

    interp = EnvInterp(tokens)
    val = interp.parse_expr()
    if interp.i != len(tokens):
        raise SyntaxError(f"trailing tokens in expr: {tokens[interp.i:]}")
    return val


class L1Interpreter:
    """
    Statement grammar (space-separated tokens):

      program := stmt+
      stmt    := set VAR expr
               | print expr
               | inc VAR | dec VAR | clr VAR
               | add VAR VAR | mul VAR VAR | mov VAR VAR

    Expressions reuse L0 atoms plus VAR.
    """

    def __init__(self, tokens: list[str], apis: list[str] | None = None):
        self.tokens = [t for t in tokens if t not in (BOS, EOS)]
        self.i = 0
        self.env: dict[str, int] = {v: 0 for v in VARS}
        self.outputs: list[int] = []
        self.apis = set(apis or (CORE_APIS + EXTRA_APIS))

    def peek(self) -> str | None:
        if self.i >= len(self.tokens):
            return None
        return self.tokens[self.i]

    def consume(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("unexpected end of program")
        if expected is not None and tok != expected:
            raise SyntaxError(f"expected {expected!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse_expr_tokens(self) -> list[str]:
        """Consume one expression; stop before next stmt keyword or end."""
        start = self.i
        balance = 0
        saw_atom = False
        while self.i < len(self.tokens):
            tok = self.tokens[self.i]
            if tok in self.apis and balance == 0 and saw_atom:
                break
            if tok == "(":
                balance += 1
            elif tok == ")":
                balance -= 1
                if balance < 0:
                    raise SyntaxError("unbalanced ')' in expression")
            if tok in DIGITS or tok in VARS or tok == ")":
                saw_atom = True
            self.i += 1
        expr = self.tokens[start : self.i]
        if not expr:
            raise SyntaxError("empty expression")
        return expr

    def eval_expr(self, expr_tokens: list[str]) -> int:
        return _eval_expr_with_env(expr_tokens, self.env)

    def run(self) -> L1Result:
        try:
            if not self.tokens:
                return L1Result(ok=False, error="empty program", tokens=self.tokens)
            while self.peek() is not None:
                self._run_stmt()
            return L1Result(
                ok=True,
                env=dict(self.env),
                outputs=list(self.outputs),
                tokens=self.tokens,
            )
        except (SyntaxError, ValueError, KeyError) as e:
            return L1Result(
                ok=False,
                env=dict(self.env),
                outputs=list(self.outputs),
                error=str(e),
                tokens=self.tokens,
            )

    def _need_var(self, var: str) -> None:
        if var not in VARS:
            raise SyntaxError(f"expected var, got {var!r}")

    def _run_stmt(self) -> None:
        op = self.consume()
        if op not in self.apis:
            raise SyntaxError(f"unknown or disabled API: {op!r}")

        if op == "set":
            var = self.consume()
            self._need_var(var)
            expr = self.parse_expr_tokens()
            self.env[var] = self.eval_expr(expr)
        elif op == "print":
            expr = self.parse_expr_tokens()
            self.outputs.append(self.eval_expr(expr))
        elif op == "inc":
            var = self.consume()
            self._need_var(var)
            self.env[var] = self.env[var] + 1
        elif op == "dec":
            var = self.consume()
            self._need_var(var)
            self.env[var] = self.env[var] - 1
        elif op == "clr":
            var = self.consume()
            self._need_var(var)
            self.env[var] = 0
        elif op in ("add", "mul", "mov"):
            a = self.consume()
            b = self.consume()
            self._need_var(a)
            self._need_var(b)
            if op == "add":
                self.env[a] = self.env[a] + self.env[b]
            elif op == "mul":
                self.env[a] = self.env[a] * self.env[b]
            else:
                self.env[a] = self.env[b]
        elif op == "neg":
            var = self.consume()
            self._need_var(var)
            self.env[var] = -self.env[var]
        elif op == "swap":
            a = self.consume()
            b = self.consume()
            self._need_var(a)
            self._need_var(b)
            self.env[a], self.env[b] = self.env[b], self.env[a]
        else:
            raise SyntaxError(f"unimplemented API: {op!r}")


def evaluate_l1(tokens: list[str], api_k: int = 8) -> L1Result:
    return L1Interpreter(tokens, apis=enabled_apis(api_k)).run()


def _emit_stmt(op: str, variables: list[str], rng: random.Random, cfg: L1Config) -> list[str]:
    if op == "set":
        var = rng.choice(variables)
        if rng.random() < 0.4:
            expr = [rng.choice(DIGITS)]
        elif rng.random() < 0.5:
            v = rng.choice(variables)
            if rng.random() < 0.5:
                expr = [v]
            else:
                expr = [v, rng.choice(["+", "-"]), rng.choice(DIGITS)]
        else:
            expr = generate_l0(rng, max_depth=cfg.expr_depth)
        return ["set", var] + expr
    if op == "print":
        if rng.random() < 0.5:
            expr = [rng.choice(variables)]
        else:
            expr = generate_l0(rng, max_depth=max(1, cfg.expr_depth - 1))
        return ["print"] + expr
    if op in ("inc", "dec", "clr"):
        return [op, rng.choice(variables)]
    if op in ("add", "mul", "mov", "swap"):
        return [op, rng.choice(variables), rng.choice(variables)]
    if op == "neg":
        return [op, rng.choice(variables)]
    raise ValueError(f"unhandled API in generator: {op}")


def generate_l1(cfg: L1Config, rng: random.Random | None = None) -> list[str]:
    """Sample a guaranteed-executable L1 program (token list, no BOS/EOS)."""
    rng = rng or random.Random(cfg.seed)
    apis = cfg.apis
    variables = cfg.vars
    n_stmts = rng.randint(cfg.min_stmts, cfg.max_stmts)

    for _attempt in range(20):
        tokens: list[str] = []
        for _ in range(n_stmts):
            tokens.extend(_emit_stmt(rng.choice(apis), variables, rng, cfg))
        result = L1Interpreter(tokens, apis=apis).run()
        if result.ok:
            return tokens

    # Fallback: minimal valid program
    v = variables[0]
    return ["set", v, "1", "print", v]


def _emit_stmt_simple(op: str, variables: list[str], rng: random.Random) -> list[str]:
    """Deterministic-ish simple statements for hold-out combos."""
    if op == "set":
        return ["set", rng.choice(variables), rng.choice(DIGITS)]
    if op == "print":
        return ["print", rng.choice(variables)]
    if op in ("inc", "dec", "clr"):
        return [op, rng.choice(variables)]
    if op in ("add", "mul", "mov", "swap"):
        return [op, rng.choice(variables), rng.choice(variables)]
    if op == "neg":
        return [op, rng.choice(variables)]
    raise ValueError(op)


def generate_l1_holdout_combo(
    cfg: L1Config,
    api_a: str,
    api_b: str,
    rng: random.Random,
) -> list[str]:
    """
    Force a program that uses both api_a and api_b (composition hold-out target).
    """
    assert api_a in cfg.apis and api_b in cfg.apis
    variables = cfg.vars

    for _attempt in range(20):
        tokens: list[str] = []
        # Always start with a set so env is non-trivial
        tokens.extend(_emit_stmt_simple("set", variables, rng))
        order = [api_a, api_b]
        rng.shuffle(order)
        for op in order:
            tokens.extend(_emit_stmt_simple(op, variables, rng))
        extra = rng.randint(0, max(0, cfg.max_stmts - 3))
        for _ in range(extra):
            tokens.extend(_emit_stmt_simple(rng.choice(cfg.apis), variables, rng))
        result = L1Interpreter(tokens, apis=cfg.apis).run()
        if result.ok and api_a in tokens and api_b in tokens:
            return tokens

    # Guaranteed fallback
    v = variables[0]
    tokens = ["set", v, "3"]
    for op in (api_a, api_b):
        tokens.extend(_emit_stmt_simple(op, variables, rng))
    return tokens
