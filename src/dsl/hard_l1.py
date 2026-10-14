"""Hard L1 freeze constants + procedural generators (template wrappers live elsewhere)."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .l1 import L1Config, L1Interpreter, _emit_stmt_simple
from .tokens import DIGITS


# --- Freeze defaults (code constants — do NOT conflate with gate knobs) ---
HARD_API_K = 6
HARD_HOLDOUT_PAIR = ("add", "mul")  # ordered: some add before some mul
FREEZE_N_VARS = 6
FREEZE_MIN_STMTS = 5
FREEZE_MAX_STMTS = 9
FREEZE_EXPR_DEPTH = 3
FREEZE_MIN_DEPS = 3
# EXTRA order for k≤8: inc, dec, add, mul, clr, mov (see tokens.EXTRA_APIS)

# Experimental knobs used by prior difficulty-gate runs (logs/manifest only)
GATE_KNOBS = {
    "min_stmts": 10,
    "max_stmts": 16,
    "min_deps": 7,
    "min_distinct_apis": 4,
    "note": "historical difficulty-gate settings; not freeze defaults",
}


@dataclass
class HardGenKnobs:
    """Runtime knobs; defaults match freeze. Gate runs override explicitly."""

    min_stmts: int = FREEZE_MIN_STMTS
    max_stmts: int = FREEZE_MAX_STMTS
    n_vars: int = FREEZE_N_VARS
    expr_depth: int = FREEZE_EXPR_DEPTH
    min_deps: int = FREEZE_MIN_DEPS
    min_distinct_apis: int = 0  # 0 = off (freeze); gate used 4

    def to_dict(self) -> dict:
        return asdict(self)


def hard_l1_config(
    seed: int | None = None,
    knobs: HardGenKnobs | None = None,
) -> L1Config:
    """Frozen hard-L1 config; optional knobs override stmt/var ranges only."""
    k = knobs or HardGenKnobs()
    return L1Config(
        api_k=HARD_API_K,
        n_vars=k.n_vars,
        min_stmts=k.min_stmts,
        max_stmts=k.max_stmts,
        expr_depth=k.expr_depth,
        seed=seed,
    )


def has_ordered_api_pair(tokens: list[str], api_a: str, api_b: str) -> bool:
    """True iff some statement-opcode A appears before some statement-opcode B."""
    idx_a = [i for i, t in enumerate(tokens) if t == api_a]
    idx_b = [i for i, t in enumerate(tokens) if t == api_b]
    if not idx_a or not idx_b:
        return False
    return any(a < b for a in idx_a for b in idx_b)


def count_cross_stmt_deps(tokens: list[str], apis: list[str]) -> int:
    """Count stmts that read a var written by an earlier stmt."""
    api_set = set(apis)
    writes: set[str] = set()
    deps = 0
    i = 0
    n = len(tokens)
    while i < n:
        op = tokens[i]
        if op not in api_set:
            i += 1
            continue
        i += 1
        if op == "set":
            if i >= n:
                break
            dest = tokens[i]
            i += 1
            expr: list[str] = []
            while i < n and tokens[i] not in api_set:
                expr.append(tokens[i])
                i += 1
            if any(t in writes for t in expr):
                deps += 1
            writes.add(dest)
        elif op == "print":
            expr = []
            while i < n and tokens[i] not in api_set:
                expr.append(tokens[i])
                i += 1
            if any(t in writes for t in expr):
                deps += 1
        elif op in ("inc", "dec", "clr", "neg"):
            if i >= n:
                break
            v = tokens[i]
            i += 1
            if v in writes:
                deps += 1
            writes.add(v)
        elif op in ("add", "mul", "mov", "swap"):
            if i + 1 >= n:
                break
            a, b = tokens[i], tokens[i + 1]
            i += 2
            if a in writes or b in writes:
                deps += 1
            writes.add(a)
            if op == "swap":
                writes.add(b)
        else:
            if i < n:
                i += 1
    return deps


def distinct_apis_used(tokens: list[str], apis: list[str]) -> int:
    api_set = set(apis)
    return len({t for t in tokens if t in api_set})


# Back-compat alias used by older imports
_distinct_apis_used = distinct_apis_used


def accept_hard_program(
    tokens: list[str],
    *,
    apis: list[str],
    api_k: int = HARD_API_K,
    min_deps: int = FREEZE_MIN_DEPS,
    min_distinct_apis: int = 0,
    require_ordered_holdout: bool | None = None,
    forbid_ordered_holdout: bool = False,
    holdout_pair: tuple[str, str] = HARD_HOLDOUT_PAIR,
) -> tuple[bool, str]:
    """
    Shared acceptance gate: interpreter ok + deps (+ optional hold-out rules).

    require_ordered_holdout=True  → must contain ordered A∘B (hold-out pool)
    forbid_ordered_holdout=True   → must NOT contain ordered A∘B (train/IID)
    """
    from .l1 import evaluate_l1

    if not evaluate_l1(tokens, api_k=api_k).ok:
        return False, "interpreter"
    if count_cross_stmt_deps(tokens, apis) < min_deps:
        return False, "deps"
    if min_distinct_apis > 0 and distinct_apis_used(tokens, apis) < min_distinct_apis:
        return False, "distinct_apis"
    a, b = holdout_pair
    ordered = has_ordered_api_pair(tokens, a, b)
    if forbid_ordered_holdout and ordered:
        return False, "ordered_holdout_forbidden"
    if require_ordered_holdout and not ordered:
        return False, "ordered_holdout_missing"
    return True, "ok"


def _n_ops(tokens: list[str], apis: list[str] | set[str]) -> int:
    api_set = set(apis)
    return sum(1 for t in tokens if t in api_set)


def _emit_op(
    tokens: list[str],
    defined: list[str],
    op: str,
    variables: list[str],
    rng: random.Random,
) -> None:
    """Append one statement; mutates tokens/defined."""
    src = rng.choice(defined) if defined else rng.choice(variables)
    dst = rng.choice(variables)
    if op == "set":
        if defined and rng.random() < 0.85:
            a = rng.choice(defined)
            if rng.random() < 0.5:
                expr = [a, rng.choice(["+", "-", "*"]), rng.choice(DIGITS)]
            else:
                b = rng.choice(defined)
                expr = ["(", a, rng.choice(["+", "-"]), b, ")"]
            tokens.extend(["set", dst] + expr)
        else:
            tokens.extend(["set", dst, rng.choice(DIGITS)])
        if dst not in defined:
            defined.append(dst)
    elif op == "print":
        if rng.random() < 0.4 and defined:
            tokens.extend(["print", rng.choice(defined), "+", rng.choice(DIGITS)])
        else:
            tokens.extend(["print", src])
    elif op in ("inc", "dec", "clr"):
        tokens.extend([op, src])
        if src not in defined:
            defined.append(src)
    elif op in ("add", "mul", "mov"):
        a = rng.choice(defined) if defined else rng.choice(variables)
        b = rng.choice(defined) if defined else rng.choice(variables)
        tokens.extend([op, a, b])
        if a not in defined:
            defined.append(a)
    else:
        tokens.extend(_emit_stmt_simple(op, variables, rng))


def generate_hard_l1(
    cfg: L1Config | None = None,
    rng: random.Random | None = None,
    min_deps: int = FREEZE_MIN_DEPS,
    min_distinct_apis: int = 0,
    forbid_ordered_holdout: bool = False,
    holdout_pair: tuple[str, str] = HARD_HOLDOUT_PAIR,
    salt: int | None = None,
) -> list[str]:
    """
    Procedural hard-L1 sampler (fallback behind template front-end).

    When forbid_ordered_holdout=True, never emit opcode B after opcode A for the
    hold-out pair (train/IID). salt bumps entropy for uniqueness under gate knobs.
    """
    cfg = cfg or hard_l1_config()
    rng = rng or random.Random(cfg.seed)
    if salt is not None:
        rng = random.Random(rng.randint(0, 2**30) ^ (salt * 2654435761))
    apis = cfg.apis
    variables = cfg.vars
    api_a, api_b = holdout_pair

    for _attempt in range(80):
        tokens: list[str] = []
        defined: list[str] = []
        n_stmts = rng.randint(cfg.min_stmts, cfg.max_stmts)
        saw_a = False

        for _ in range(rng.randint(2, 3)):
            v = rng.choice(variables)
            tokens.extend(["set", v, rng.choice(DIGITS)])
            if v not in defined:
                defined.append(v)

        # Prefer a diverse schedule covering ≥ min_distinct_apis
        schedule = list(apis)
        rng.shuffle(schedule)
        while len(schedule) < n_stmts:
            schedule.append(rng.choice(apis))
        schedule = schedule[: max(0, n_stmts - _n_ops(tokens, apis))]

        for op in schedule:
            if forbid_ordered_holdout and op == api_b and saw_a:
                # Avoid ordered A∘B — use a safe alternative
                op = rng.choice([x for x in apis if x not in (api_b,)] or ["inc", "dec", "print"])
            _emit_op(tokens, defined, op, variables, rng)
            if op == api_a:
                saw_a = True

        # Pad until stmt / deps / distinct targets met
        pad_guard = 0
        while pad_guard < 40:
            ok, reason = accept_hard_program(
                tokens,
                apis=apis,
                min_deps=min_deps,
                min_distinct_apis=min_distinct_apis,
                forbid_ordered_holdout=forbid_ordered_holdout,
                holdout_pair=holdout_pair,
            )
            if ok:
                return tokens
            # Choose fill op based on missing constraint
            if reason == "ordered_holdout_forbidden":
                break  # rebuild
            allowed = list(apis)
            if forbid_ordered_holdout and saw_a:
                allowed = [x for x in allowed if x != api_b] or ["inc", "print", "dec"]
            op = rng.choice(allowed)
            _emit_op(tokens, defined, op, variables, rng)
            if op == api_a:
                saw_a = True
            pad_guard += 1

        ok, _ = accept_hard_program(
            tokens,
            apis=apis,
            min_deps=min_deps,
            min_distinct_apis=min_distinct_apis,
            forbid_ordered_holdout=forbid_ordered_holdout,
            holdout_pair=holdout_pair,
        )
        if ok:
            return tokens

    # Deterministic long chain meeting gate knobs without ordered add∘mul
    # (mul before add = reverse order, allowed in train)
    vs = variables[:6]
    a, b, c = vs[0], vs[1], vs[2]
    tokens = [
        "set", a, "2",
        "set", b, "5",
        "set", c, a, "+", "1",
        "mul", b, a,
        "inc", c,
        "dec", a,
        "print", b,
        "add", c, b,
        "inc", b,
        "print", c,
        "dec", b,
        "set", a, "(", c, "-", a, ")",
        "print", a,
    ]
    if salt is not None:
        tokens.extend(["inc", vs[salt % len(vs)]])
    return tokens


def generate_hard_holdout_combo(
    cfg: L1Config,
    api_a: str,
    api_b: str,
    rng: random.Random,
    min_deps: int = FREEZE_MIN_DEPS,
    min_distinct_apis: int = 0,
    salt: int | None = None,
) -> list[str]:
    """Procedural ordered A∘B hold-out sampler (fallback)."""
    assert api_a in cfg.apis and api_b in cfg.apis
    if salt is not None:
        rng = random.Random(rng.randint(0, 2**30) ^ (salt * 40503))
    variables = cfg.vars
    apis = cfg.apis
    target_stmts = rng.randint(cfg.min_stmts, cfg.max_stmts)

    for _attempt in range(60):
        tokens: list[str] = []
        defined: list[str] = []
        v0 = rng.choice(variables)
        tokens.extend(["set", v0, rng.choice(DIGITS)])
        defined.append(v0)
        v1 = rng.choice(variables)
        tokens.extend(["set", v1, v0, "+", rng.choice(DIGITS)])
        if v1 not in defined:
            defined.append(v1)

        _emit_op(tokens, defined, api_a, variables, rng)
        for _ in range(rng.randint(2, 5)):
            op = rng.choice([x for x in apis if x not in (api_a, api_b)] or apis)
            _emit_op(tokens, defined, op, variables, rng)
        _emit_op(tokens, defined, api_b, variables, rng)

        pad = 0
        while _n_ops(tokens, apis) < target_stmts and pad < 30:
            op = rng.choice(apis)
            _emit_op(tokens, defined, op, variables, rng)
            pad += 1

        ok, _ = accept_hard_program(
            tokens,
            apis=apis,
            min_deps=min_deps,
            min_distinct_apis=min_distinct_apis,
            require_ordered_holdout=True,
            holdout_pair=(api_a, api_b),
        )
        if ok:
            return tokens

    a, b, c = variables[0], variables[1], variables[2]
    tokens = [
        "set", a, "4",
        "set", b, a, "+", "1",
        "set", c, "2",
        "add", b, a,
        "inc", c,
        "dec", a,
        "print", b,
        "mul", b, a,
        "inc", b,
        "add", c, b,
        "print", c,
        "dec", b,
        "print", a,
    ]
    if salt is not None:
        tokens.extend(["inc", variables[salt % len(variables)]])
    return tokens


def force_unique_hard_program(
    seen: set[tuple[str, ...]],
    cfg: L1Config,
    rng: random.Random,
    *,
    holdout: bool,
    min_deps: int,
    min_distinct_apis: int = 0,
    holdout_pair: tuple[str, str] = HARD_HOLDOUT_PAIR,
    max_tries: int = 200,
) -> list[str] | None:
    """
    Keep sampling procedural programs (with salt) until a new unique key appears.
    Returns None only if exhausted (should be rare).
    """
    api_a, api_b = holdout_pair
    for i in range(max_tries):
        if holdout:
            toks = generate_hard_holdout_combo(
                cfg, api_a, api_b, rng,
                min_deps=min_deps,
                min_distinct_apis=min_distinct_apis,
                salt=i + len(seen) * 17,
            )
            ok, _ = accept_hard_program(
                toks, apis=cfg.apis, min_deps=min_deps,
                min_distinct_apis=min_distinct_apis,
                require_ordered_holdout=True, holdout_pair=holdout_pair,
            )
        else:
            toks = generate_hard_l1(
                cfg, rng, min_deps=min_deps,
                min_distinct_apis=min_distinct_apis,
                forbid_ordered_holdout=True,
                holdout_pair=holdout_pair,
                salt=i + len(seen) * 31,
            )
            ok, _ = accept_hard_program(
                toks, apis=cfg.apis, min_deps=min_deps,
                min_distinct_apis=min_distinct_apis,
                forbid_ordered_holdout=True, holdout_pair=holdout_pair,
            )
        if not ok:
            continue
        key = tuple(toks)
        if key not in seen:
            return toks
    return None
