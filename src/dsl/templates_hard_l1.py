"""
Cadmus-style template sampling front-end for hard L1.

Templates propose candidate token lists; acceptance remains
interpreter-ok + ordered hold-out + deps gates (see hard_l1.accept_hard_program).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .hard_l1 import (
    HARD_API_K,
    HARD_HOLDOUT_PAIR,
    FREEZE_MIN_DEPS,
    HardGenKnobs,
    accept_hard_program,
    generate_hard_l1,
    generate_hard_holdout_combo,
    hard_l1_config,
    _emit_op,
    _n_ops,
    has_ordered_api_pair,
)
from .l1 import L1Config, evaluate_l1, generate_l1
from .tokens import DIGITS, enabled_apis


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    family: str  # pipeline | fan-in | reuse | holdout-combo
    description: str


TEMPLATES: list[TemplateSpec] = [
    TemplateSpec("pipe.chain_set_add_inc_print", "pipeline", "Linear write→combine→bump→print"),
    TemplateSpec("pipe.dec_mul_print", "pipeline", "set→dec→mul→print chain"),
    TemplateSpec("pipe.gate_long_mul_first", "pipeline", "Gate-length chain; mul before add (train-safe)"),
    TemplateSpec("fanin.two_src_add", "fan-in", "Two independent sets fan into add"),
    TemplateSpec("fanin.three_src_mul", "fan-in", "Three sources; mul then print"),
    TemplateSpec("fanin.gate_wide", "fan-in", "Gate-length multi-source fan-in"),
    TemplateSpec("reuse.same_var_loopish", "reuse", "Heavy reuse of one accumulator var"),
    TemplateSpec("reuse.dual_pingpong", "reuse", "Two vars alternately updated"),
    TemplateSpec("reuse.gate_triple", "reuse", "Gate-length three-var reuse (train-safe)"),
    TemplateSpec("holdout.add_then_mul", "holdout-combo", "Ordered add∘mul with deps"),
    TemplateSpec("holdout.add_filler_mul", "holdout-combo", "add, filler, then mul"),
    TemplateSpec("holdout.gate_long", "holdout-combo", "Gate-length ordered add∘mul"),
]


def _pick_vars(rng: random.Random, n: int, pool: list[str]) -> list[str]:
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]


def _digit(rng: random.Random) -> str:
    return rng.choice(DIGITS)


def _tpl_pipeline_chain(rng: random.Random, variables: list[str]) -> list[str]:
    a, b, c = _pick_vars(rng, 3, variables)
    return [
        "set", a, _digit(rng),
        "set", b, a, "+", _digit(rng),
        "add", b, a,
        "inc", b,
        "set", c, b, "-", _digit(rng),
        "print", c,
    ]


def _tpl_pipeline_dec_mul(rng: random.Random, variables: list[str]) -> list[str]:
    a, b = _pick_vars(rng, 2, variables)
    return [
        "set", a, _digit(rng),
        "set", b, _digit(rng),
        "dec", a,
        "mul", b, a,
        "inc", b,
        "print", a,
        "print", "(", b, "+", a, ")",
    ]


def _tpl_fanin_two(rng: random.Random, variables: list[str]) -> list[str]:
    a, b, c = _pick_vars(rng, 3, variables)
    return [
        "set", a, _digit(rng),
        "set", b, _digit(rng),
        "set", c, "0",
        "add", c, a,
        "add", c, b,
        "inc", c,
        "print", c,
    ]


def _tpl_fanin_three(rng: random.Random, variables: list[str]) -> list[str]:
    a, b, c, d = _pick_vars(rng, 4, variables)
    return [
        "set", a, _digit(rng),
        "set", b, _digit(rng),
        "set", c, _digit(rng),
        "set", d, a,
        "mul", d, b,
        "add", d, c,
        "dec", d,
        "print", d,
    ]


def _tpl_reuse_accum(rng: random.Random, variables: list[str]) -> list[str]:
    a, b = _pick_vars(rng, 2, variables)
    return [
        "set", a, _digit(rng),
        "inc", a,
        "add", a, a,
        "dec", a,
        "set", b, a, "+", _digit(rng),
        "add", a, b,
        "print", a,
        "print", b,
    ]


def _tpl_reuse_pingpong(rng: random.Random, variables: list[str]) -> list[str]:
    a, b = _pick_vars(rng, 2, variables)
    return [
        "set", a, _digit(rng),
        "set", b, _digit(rng),
        "add", a, b,
        "add", b, a,
        "inc", a,
        "dec", b,
        "mul", a, b,
        "print", a,
        "print", b,
    ]


def _tpl_holdout_add_mul(rng: random.Random, variables: list[str]) -> list[str]:
    a, b, c = _pick_vars(rng, 3, variables)
    return [
        "set", a, _digit(rng),
        "set", b, a, "+", _digit(rng),
        "add", b, a,
        "inc", a,
        "mul", b, a,
        "set", c, b, "-", "1",
        "print", c,
    ]


def _tpl_holdout_filler(rng: random.Random, variables: list[str]) -> list[str]:
    a, b = _pick_vars(rng, 2, variables)
    return [
        "set", a, _digit(rng),
        "set", b, _digit(rng),
        "add", a, b,
        "dec", b,
        "inc", a,
        "print", a,
        "mul", a, b,
        "print", a,
    ]


def _tpl_pipe_gate_mul_first(rng: random.Random, variables: list[str]) -> list[str]:
    """Long train-safe chain: mul appears before add (not ordered add∘mul)."""
    a, b, c, d = _pick_vars(rng, 4, variables)
    return [
        "set", a, _digit(rng),
        "set", b, _digit(rng),
        "set", c, a, "+", _digit(rng),
        "mul", c, b,
        "inc", a,
        "dec", b,
        "print", c,
        "set", d, "(", c, "-", a, ")",
        "mul", d, a,
        "inc", d,
        "add", d, b,
        "print", d,
        "dec", a,
        "print", b,
    ]


def _tpl_fanin_gate_wide(rng: random.Random, variables: list[str]) -> list[str]:
    a, b, c, d = _pick_vars(rng, 4, variables)
    return [
        "set", a, _digit(rng),
        "set", b, _digit(rng),
        "set", c, _digit(rng),
        "set", d, "0",
        "mul", d, a,  # d=0*a
        "inc", d,
        "add", d, a,
        "add", d, b,
        "add", d, c,
        "dec", d,
        "inc", a,
        "print", d,
        "print", a,
        "dec", b,
        "print", c,
    ]


def _tpl_reuse_gate_triple(rng: random.Random, variables: list[str]) -> list[str]:
    a, b, c = _pick_vars(rng, 3, variables)
    return [
        "set", a, _digit(rng),
        "set", b, _digit(rng),
        "set", c, a,
        "inc", a,
        "inc", a,
        "add", a, b,
        "dec", b,
        "add", c, a,
        "print", a,
        "print", b,
        "inc", c,
        "dec", c,
        "print", c,
        "add", a, c,
        "print", a,
    ]


def _tpl_holdout_gate_long(rng: random.Random, variables: list[str]) -> list[str]:
    a, b, c = _pick_vars(rng, 3, variables)
    return [
        "set", a, _digit(rng),
        "set", b, a, "+", _digit(rng),
        "set", c, _digit(rng),
        "inc", c,
        "add", b, a,
        "dec", a,
        "print", b,
        "inc", b,
        "mul", b, a,
        "add", c, b,
        "print", c,
        "dec", b,
        "print", a,
        "inc", a,
        "print", b,
    ]


_BUILDERS: dict[str, Callable[[random.Random, list[str]], list[str]]] = {
    "pipe.chain_set_add_inc_print": _tpl_pipeline_chain,
    "pipe.dec_mul_print": _tpl_pipeline_dec_mul,
    "pipe.gate_long_mul_first": _tpl_pipe_gate_mul_first,
    "fanin.two_src_add": _tpl_fanin_two,
    "fanin.three_src_mul": _tpl_fanin_three,
    "fanin.gate_wide": _tpl_fanin_gate_wide,
    "reuse.same_var_loopish": _tpl_reuse_accum,
    "reuse.dual_pingpong": _tpl_reuse_pingpong,
    "reuse.gate_triple": _tpl_reuse_gate_triple,
    "holdout.add_then_mul": _tpl_holdout_add_mul,
    "holdout.add_filler_mul": _tpl_holdout_filler,
    "holdout.gate_long": _tpl_holdout_gate_long,
}


def pad_toward_knobs(
    tokens: list[str],
    rng: random.Random,
    *,
    apis: list[str],
    variables: list[str],
    min_stmts: int,
    min_deps: int,
    min_distinct_apis: int,
    holdout: bool,
    holdout_pair: tuple[str, str],
    max_pad: int = 24,
) -> list[str]:
    """Append safe stmts so short templates can pass gate knobs."""
    api_a, api_b = holdout_pair
    out = list(tokens)
    defined = [t for t in out if t in variables]
    if not defined:
        defined = [variables[0]]
    saw_a = api_a in out

    for _ in range(max_pad):
        ok, reason = accept_hard_program(
            out,
            apis=apis,
            min_deps=min_deps,
            min_distinct_apis=min_distinct_apis,
            require_ordered_holdout=True if holdout else None,
            forbid_ordered_holdout=(not holdout),
            holdout_pair=holdout_pair,
        )
        if ok and _n_ops(out, apis) >= min_stmts:
            return out
        if holdout and reason == "ordered_holdout_missing":
            _emit_op(out, defined, api_a, variables, rng)
            _emit_op(out, defined, api_b, variables, rng)
            continue
        allowed = list(apis)
        if not holdout and saw_a:
            allowed = [x for x in allowed if x != api_b] or ["inc", "dec", "print"]
        op = rng.choice(allowed)
        _emit_op(out, defined, op, variables, rng)
        if op == api_a:
            saw_a = True
    return out


def instantiate(template_id: str, rng: random.Random, variables: list[str]) -> list[str]:
    if template_id not in _BUILDERS:
        raise KeyError(f"unknown template_id: {template_id}")
    return _BUILDERS[template_id](rng, variables)


def templates_for_family(family: str) -> list[TemplateSpec]:
    return [t for t in TEMPLATES if t.family == family]


def choose_template(rng: random.Random, *, holdout: bool) -> TemplateSpec:
    if holdout:
        pool = templates_for_family("holdout-combo")
    else:
        pool = [t for t in TEMPLATES if t.family != "holdout-combo"]
    return rng.choice(pool)


@dataclass
class SampleStats:
    proposals: int = 0
    accepts: int = 0
    rejects: dict[str, int] | None = None
    template_accepts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.rejects is None:
            self.rejects = {}
        if self.template_accepts is None:
            self.template_accepts = {}

    def reject(self, reason: str) -> None:
        assert self.rejects is not None
        self.rejects[reason] = self.rejects.get(reason, 0) + 1

    def accept(self, template_id: str) -> None:
        self.accepts += 1
        assert self.template_accepts is not None
        self.template_accepts[template_id] = self.template_accepts.get(template_id, 0) + 1

    @property
    def reject_rate(self) -> float:
        if self.proposals == 0:
            return 0.0
        return 1.0 - self.accepts / self.proposals

    def to_dict(self) -> dict:
        return {
            "proposals": self.proposals,
            "accepts": self.accepts,
            "reject_rate": round(self.reject_rate, 4),
            "rejects": dict(self.rejects or {}),
            "template_accepts": dict(self.template_accepts or {}),
        }


def sample_with_templates(
    rng: random.Random,
    *,
    holdout: bool = False,
    cfg: L1Config | None = None,
    knobs: HardGenKnobs | None = None,
    min_deps: int | None = None,
    min_distinct_apis: int = 0,
    holdout_pair: tuple[str, str] = HARD_HOLDOUT_PAIR,
    max_tries: int = 40,
    fallback_procedural: bool = True,
    stats: SampleStats | None = None,
    salt: int | None = None,
    prefer_procedural: bool = False,
) -> tuple[list[str], str]:
    """
    Template candidate → pad toward knobs → acceptance gates.
    Falls back to hold-out-aware procedural (never returns ungated emergency).
    """
    knobs = knobs or HardGenKnobs()
    cfg = cfg or hard_l1_config(knobs=knobs)
    deps = FREEZE_MIN_DEPS if min_deps is None else min_deps
    apis = cfg.apis
    variables = cfg.vars
    stats = stats or SampleStats()
    api_a, api_b = holdout_pair
    if salt is not None:
        rng = random.Random(rng.randint(0, 2**30) ^ salt)

    # Optional early procedural (hang escape / caller request). Under tight
    # gate knobs alone we still try padded templates first (~35% procedural mix).
    use_proc_first = prefer_procedural or (
        min_distinct_apis >= 4 and deps >= 7 and rng.random() < 0.35
    )
    if use_proc_first:
        stats.proposals += 1
        if holdout:
            tokens = generate_hard_holdout_combo(
                cfg, api_a, api_b, rng, min_deps=deps,
                min_distinct_apis=min_distinct_apis, salt=salt,
            )
            ok, reason = accept_hard_program(
                tokens, apis=apis, min_deps=deps,
                min_distinct_apis=min_distinct_apis,
                require_ordered_holdout=True, holdout_pair=holdout_pair,
            )
        else:
            tokens = generate_hard_l1(
                cfg, rng, min_deps=deps, min_distinct_apis=min_distinct_apis,
                forbid_ordered_holdout=True, holdout_pair=holdout_pair, salt=salt,
            )
            ok, reason = accept_hard_program(
                tokens, apis=apis, min_deps=deps,
                min_distinct_apis=min_distinct_apis,
                forbid_ordered_holdout=True, holdout_pair=holdout_pair,
            )
        if ok:
            stats.accept("procedural_primary")
            return tokens, "procedural_primary"
        stats.reject(reason)

    for _ in range(max_tries):
        spec = choose_template(rng, holdout=holdout)
        stats.proposals += 1
        tokens = instantiate(spec.template_id, rng, variables)
        tokens = pad_toward_knobs(
            tokens, rng, apis=apis, variables=variables,
            min_stmts=knobs.min_stmts, min_deps=deps,
            min_distinct_apis=min_distinct_apis, holdout=holdout,
            holdout_pair=holdout_pair,
        )
        ok, reason = accept_hard_program(
            tokens,
            apis=apis,
            min_deps=deps,
            min_distinct_apis=min_distinct_apis,
            require_ordered_holdout=True if holdout else None,
            forbid_ordered_holdout=(not holdout),
            holdout_pair=holdout_pair,
        )
        if ok:
            stats.accept(spec.template_id)
            return tokens, spec.template_id
        stats.reject(reason)

    if fallback_procedural:
        for attempt in range(20):
            stats.proposals += 1
            if holdout:
                tokens = generate_hard_holdout_combo(
                    cfg, api_a, api_b, rng, min_deps=deps,
                    min_distinct_apis=min_distinct_apis,
                    salt=(salt or 0) + attempt,
                )
                ok, reason = accept_hard_program(
                    tokens, apis=apis, min_deps=deps,
                    min_distinct_apis=min_distinct_apis,
                    require_ordered_holdout=True, holdout_pair=holdout_pair,
                )
            else:
                tokens = generate_hard_l1(
                    cfg, rng, min_deps=deps, min_distinct_apis=min_distinct_apis,
                    forbid_ordered_holdout=True, holdout_pair=holdout_pair,
                    salt=(salt or 0) + attempt * 13,
                )
                ok, reason = accept_hard_program(
                    tokens, apis=apis, min_deps=deps,
                    min_distinct_apis=min_distinct_apis,
                    forbid_ordered_holdout=True, holdout_pair=holdout_pair,
                )
            if ok:
                stats.accept("procedural_fallback")
                return tokens, "procedural_fallback"
            stats.reject(reason)

    # Last resort: still gated procedural (may collide uniqueness upstream)
    if holdout:
        tokens = generate_hard_holdout_combo(
            cfg, api_a, api_b, rng, min_deps=max(1, deps // 2),
            min_distinct_apis=0, salt=salt,
        )
        stats.accept("procedural_relaxed_holdout")
        return tokens, "procedural_relaxed_holdout"
    tokens = generate_hard_l1(
        cfg, rng, min_deps=max(1, deps // 2), min_distinct_apis=0,
        forbid_ordered_holdout=True, holdout_pair=holdout_pair, salt=salt,
    )
    stats.accept("procedural_relaxed_iid")
    return tokens, "procedural_relaxed_iid"


def sample_mixture_curriculum(
    rng: random.Random,
    *,
    k_choices: list[int] | None = None,
    knobs: HardGenKnobs | None = None,
    min_deps: int | None = None,
    stats: SampleStats | None = None,
) -> tuple[list[str], str, int]:
    """
    Mixture/curriculum arm: sample with api_k ∈ {2,3,6}.
    Returns (tokens, template_id, api_k_used).
    """
    k_choices = k_choices or [2, 3, 6]
    knobs = knobs or HardGenKnobs()
    deps = FREEZE_MIN_DEPS if min_deps is None else min_deps
    stats = stats or SampleStats()
    k = rng.choice(k_choices)

    if k >= 6:
        tokens, tid = sample_with_templates(
            rng, holdout=False, knobs=knobs, min_deps=deps, stats=stats
        )
        return tokens, tid, HARD_API_K

    cfg = L1Config(
        api_k=k,
        n_vars=knobs.n_vars,
        min_stmts=max(2, knobs.min_stmts - 2),
        max_stmts=max(3, knobs.max_stmts - 2),
        expr_depth=knobs.expr_depth,
    )
    apis = enabled_apis(k)
    for _ in range(30):
        stats.proposals += 1
        tokens = generate_l1(cfg, rng)
        if not evaluate_l1(tokens, api_k=k).ok:
            stats.reject("interpreter")
            continue
        if "add" in apis and "mul" in apis:
            ok, reason = accept_hard_program(
                tokens,
                apis=apis,
                api_k=k,
                min_deps=min(deps, 2),
                forbid_ordered_holdout=True,
            )
            if not ok:
                stats.reject(reason)
                continue
        stats.accept(f"mixture_k{k}")
        return tokens, f"mixture_k{k}", k

    tokens = generate_l1(cfg, rng)
    stats.accept(f"mixture_k{k}_raw")
    return tokens, f"mixture_k{k}_raw", k
