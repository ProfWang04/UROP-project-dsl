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
    TemplateSpec("fanin.two_src_add", "fan-in", "Two independent sets fan into add"),
    TemplateSpec("fanin.three_src_mul", "fan-in", "Three sources; mul then print"),
    TemplateSpec("reuse.same_var_loopish", "reuse", "Heavy reuse of one accumulator var"),
    TemplateSpec("reuse.dual_pingpong", "reuse", "Two vars alternately updated"),
    TemplateSpec("holdout.add_then_mul", "holdout-combo", "Ordered add∘mul with deps"),
    TemplateSpec("holdout.add_filler_mul", "holdout-combo", "add, filler, then mul"),
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


_BUILDERS: dict[str, Callable[[random.Random, list[str]], list[str]]] = {
    "pipe.chain_set_add_inc_print": _tpl_pipeline_chain,
    "pipe.dec_mul_print": _tpl_pipeline_dec_mul,
    "fanin.two_src_add": _tpl_fanin_two,
    "fanin.three_src_mul": _tpl_fanin_three,
    "reuse.same_var_loopish": _tpl_reuse_accum,
    "reuse.dual_pingpong": _tpl_reuse_pingpong,
    "holdout.add_then_mul": _tpl_holdout_add_mul,
    "holdout.add_filler_mul": _tpl_holdout_filler,
}


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
) -> tuple[list[str], str]:
    """
    Template candidate → acceptance gates.
    Falls back to procedural generate_hard_l1 / generate_hard_holdout_combo.
    Returns (tokens, template_id_or_tag).
    """
    knobs = knobs or HardGenKnobs()
    cfg = cfg or hard_l1_config(knobs=knobs)
    deps = FREEZE_MIN_DEPS if min_deps is None else min_deps
    apis = cfg.apis
    variables = cfg.vars
    stats = stats or SampleStats()
    api_a, api_b = holdout_pair

    for _ in range(max_tries):
        spec = choose_template(rng, holdout=holdout)
        stats.proposals += 1
        tokens = instantiate(spec.template_id, rng, variables)
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
        stats.proposals += 1
        if holdout:
            tokens = generate_hard_holdout_combo(cfg, api_a, api_b, rng, min_deps=deps)
            ok, reason = accept_hard_program(
                tokens,
                apis=apis,
                min_deps=deps,
                min_distinct_apis=min_distinct_apis,
                require_ordered_holdout=True,
                holdout_pair=holdout_pair,
            )
        else:
            tokens = generate_hard_l1(
                cfg, rng, min_deps=deps, min_distinct_apis=min_distinct_apis
            )
            ok, reason = accept_hard_program(
                tokens,
                apis=apis,
                min_deps=deps,
                min_distinct_apis=min_distinct_apis,
                forbid_ordered_holdout=True,
                holdout_pair=holdout_pair,
            )
        if ok:
            stats.accept("procedural_fallback")
            return tokens, "procedural_fallback"
        stats.reject(reason)

    v = variables[0]
    if holdout:
        tokens = ["set", v, "3", "add", v, v, "inc", v, "mul", v, v, "print", v]
        stats.accept("emergency_holdout")
        return tokens, "emergency_holdout"
    tokens = ["set", v, "1", "inc", v, "print", v, "dec", v, "print", v]
    stats.accept("emergency_iid")
    return tokens, "emergency_iid"


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
