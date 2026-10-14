"""Generate L0/L1 datasets with IID and composition hold-out splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from src.dsl.l0 import generate_l0, evaluate_l0
from src.dsl.l1 import L1Config, generate_l1, generate_l1_holdout_combo, evaluate_l1
from src.dsl.hard_l1 import (
    HARD_API_K,
    HARD_HOLDOUT_PAIR,
    FREEZE_MIN_DEPS,
    GATE_KNOBS,
    HardGenKnobs,
    hard_l1_config,
    has_ordered_api_pair,
    count_cross_stmt_deps,
)
from src.dsl.templates_hard_l1 import (
    TEMPLATES,
    SampleStats,
    sample_with_templates,
    sample_mixture_curriculum,
)
from src.dsl.tokens import BOS, EOS, build_vocab


@dataclass
class DatasetSplit:
    train: list[list[str]]
    test_iid: list[list[str]]
    test_holdout: list[list[str]]
    meta: dict


def _wrap(tokens: list[str]) -> list[str]:
    return [BOS] + tokens + [EOS]


def holdout_pair_hash(pair: tuple[str, str] | list[str]) -> str:
    raw = f"{pair[0]}|{pair[1]}|ordered"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_l0_dataset(
    n_train: int,
    n_test: int = 200,
    max_depth: int = 3,
    seed: int = 0,
) -> DatasetSplit:
    rng = random.Random(seed)
    train, test = [], []
    seen: set[tuple[str, ...]] = set()

    def sample() -> list[str]:
        for _ in range(100):
            t = generate_l0(rng, max_depth=max_depth)
            key = tuple(t)
            if key not in seen and evaluate_l0(t).ok:
                seen.add(key)
                return t
        return generate_l0(rng, max_depth=1)

    for _ in range(n_train):
        train.append(_wrap(sample()))
    for _ in range(n_test):
        test.append(_wrap(sample()))

    return DatasetSplit(
        train=train,
        test_iid=test,
        test_holdout=[],
        meta={"level": "l0", "n_train": n_train, "max_depth": max_depth, "seed": seed},
    )


def generate_l1_dataset(
    n_train: int,
    api_k: int = 3,
    n_test_iid: int = 200,
    n_test_holdout: int = 200,
    seed: int = 0,
    holdout_pair: tuple[str, str] | None = None,
    hard: bool = False,
    min_deps: int | None = None,
    use_templates: bool = True,
    arm: str | None = None,  # soft | hard | mixture
    knobs: HardGenKnobs | None = None,
    gate_knobs: bool = False,
) -> DatasetSplit:
    """
    hard arm: k=6 template front-end + freeze gates.
    mixture arm: curriculum k'∈{2,3,6} for train; hold-out still k=6 ordered add∘mul.
    soft: toy both-present hold-out (legacy).
    """
    if arm is None:
        arm = "hard" if hard else "soft"
    use_hard = arm in ("hard", "mixture") or hard
    rng = random.Random(seed)
    if gate_knobs:
        knobs = HardGenKnobs(
            min_stmts=GATE_KNOBS["min_stmts"],
            max_stmts=GATE_KNOBS["max_stmts"],
            min_deps=GATE_KNOBS["min_deps"],
            min_distinct_apis=GATE_KNOBS["min_distinct_apis"],
        )
    knobs = knobs or HardGenKnobs()
    deps = knobs.min_deps if min_deps is None else min_deps

    if use_hard:
        cfg = hard_l1_config(seed=seed, knobs=knobs)
        api_k_eval = HARD_API_K
        holdout_pair = holdout_pair or HARD_HOLDOUT_PAIR
    else:
        cfg = L1Config(api_k=api_k, seed=seed)
        api_k_eval = api_k

    apis = cfg.apis
    if holdout_pair is None:
        if len(apis) >= 4:
            holdout_pair = (apis[2], apis[3])
        elif len(apis) >= 3:
            holdout_pair = (apis[1], apis[2])
        else:
            holdout_pair = (apis[0], apis[1])
    api_a, api_b = holdout_pair

    train_stats = SampleStats()
    iid_stats = SampleStats()
    holdout_stats = SampleStats()
    template_ids_train: list[str] = []

    def sample_train_like(stats: SampleStats) -> list[str] | None:
        if arm == "hard" or (hard and arm != "mixture"):
            if use_templates:
                toks, tid = sample_with_templates(
                    rng,
                    holdout=False,
                    cfg=cfg,
                    knobs=knobs,
                    min_deps=deps,
                    min_distinct_apis=knobs.min_distinct_apis,
                    holdout_pair=(api_a, api_b),
                    stats=stats,
                )
                template_ids_train.append(tid)
                return toks
            from src.dsl.hard_l1 import generate_hard_l1, accept_hard_program

            toks = generate_hard_l1(
                cfg, rng, min_deps=deps, min_distinct_apis=knobs.min_distinct_apis
            )
            ok, _ = accept_hard_program(
                toks,
                apis=apis,
                min_deps=deps,
                min_distinct_apis=knobs.min_distinct_apis,
                forbid_ordered_holdout=True,
                holdout_pair=(api_a, api_b),
            )
            return toks if ok else None

        if arm == "mixture":
            toks, tid, _k = sample_mixture_curriculum(
                rng, knobs=knobs, min_deps=deps, stats=stats
            )
            template_ids_train.append(tid)
            return toks

        # soft toy path
        toks = generate_l1(cfg, rng)
        if api_a in toks and api_b in toks:
            return None
        if not evaluate_l1(toks, api_k=api_k_eval).ok:
            return None
        return toks

    train: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    while len(train) < n_train:
        toks = sample_train_like(train_stats)
        if toks is None:
            continue
        key = tuple(toks)
        if key in seen:
            continue
        seen.add(key)
        train.append(_wrap(toks))

    test_iid: list[list[str]] = []
    while len(test_iid) < n_test_iid:
        # Mixture: IID eval on hard k=6 templates for fair N(k=6) curve
        if arm == "mixture":
            toks, _tid = sample_with_templates(
                rng,
                holdout=False,
                cfg=cfg,
                knobs=knobs,
                min_deps=deps,
                holdout_pair=(api_a, api_b),
                stats=iid_stats,
            )
        else:
            toks = sample_train_like(iid_stats)
            if toks is None:
                continue
        key = tuple(toks)
        if key in seen:
            continue
        seen.add(key)
        test_iid.append(_wrap(toks))

    test_holdout: list[list[str]] = []
    while len(test_holdout) < n_test_holdout:
        if use_hard:
            toks, _tid = sample_with_templates(
                rng,
                holdout=True,
                cfg=cfg,
                knobs=knobs,
                min_deps=deps,
                holdout_pair=(api_a, api_b),
                stats=holdout_stats,
            )
        else:
            toks = generate_l1_holdout_combo(cfg, api_a, api_b, rng)
        key = tuple(toks)
        if key in seen:
            continue
        if use_hard:
            if not has_ordered_api_pair(toks, api_a, api_b):
                continue
        elif not (api_a in toks and api_b in toks):
            continue
        if not evaluate_l1(toks, api_k=api_k_eval).ok:
            continue
        seen.add(key)
        test_holdout.append(_wrap(toks))

    meta = {
        "level": "l1",
        "arm": arm,
        "hard": use_hard,
        "use_templates": use_templates and use_hard,
        "n_train": n_train,
        "api_k": api_k_eval if use_hard else api_k,
        "apis": enabled_apis_safe(api_k_eval if use_hard else api_k),
        "holdout_pair": [api_a, api_b],
        "holdout_pair_hash": holdout_pair_hash((api_a, api_b)),
        "holdout_mode": "ordered_A_then_B" if use_hard else "both_present",
        "freeze_defaults": {
            "HARD_API_K": HARD_API_K,
            "HARD_HOLDOUT_PAIR": list(HARD_HOLDOUT_PAIR),
            "n_vars": 6,
            "min_stmts": 5,
            "max_stmts": 9,
            "expr_depth": 3,
            "min_deps": FREEZE_MIN_DEPS,
        },
        "experiment_knobs": knobs.to_dict(),
        "gate_knobs_reference": GATE_KNOBS,
        "min_deps": deps,
        "cfg": {
            "n_vars": cfg.n_vars,
            "min_stmts": cfg.min_stmts,
            "max_stmts": cfg.max_stmts,
            "expr_depth": cfg.expr_depth,
        },
        "template_catalog": [
            {"id": t.template_id, "family": t.family} for t in TEMPLATES
        ],
        "sample_stats": {
            "train": train_stats.to_dict(),
            "test_iid": iid_stats.to_dict(),
            "test_holdout": holdout_stats.to_dict(),
        },
        "seed": seed,
    }

    return DatasetSplit(
        train=train,
        test_iid=test_iid,
        test_holdout=test_holdout,
        meta=meta,
    )


def enabled_apis_safe(api_k: int) -> list[str]:
    from src.dsl.tokens import enabled_apis

    return enabled_apis(api_k)


def generate_dataset(
    level: str = "l1",
    n_train: int = 1000,
    api_k: int = 3,
    seed: int = 0,
    hard: bool = False,
    arm: str | None = None,
    **kwargs,
) -> DatasetSplit:
    if level == "l0":
        return generate_l0_dataset(n_train=n_train, seed=seed, **kwargs)
    return generate_l1_dataset(
        n_train=n_train, api_k=api_k, seed=seed, hard=hard, arm=arm, **kwargs
    )


def save_dataset(split: DatasetSplit, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("train", split.train),
        ("test_iid", split.test_iid),
        ("test_holdout", split.test_holdout),
    ):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for seq in data:
                f.write(json.dumps({"tokens": seq}) + "\n")
    (out_dir / "meta.json").write_text(json.dumps(split.meta, indent=2), encoding="utf-8")
    # Manifest: freeze vs knobs + sampling stats (explicit filename)
    manifest = {
        "freeze_defaults": split.meta.get("freeze_defaults"),
        "experiment_knobs": split.meta.get("experiment_knobs"),
        "gate_knobs_reference": split.meta.get("gate_knobs_reference"),
        "holdout_pair": split.meta.get("holdout_pair"),
        "holdout_pair_hash": split.meta.get("holdout_pair_hash"),
        "arm": split.meta.get("arm"),
        "seed": split.meta.get("seed"),
        "template_catalog": split.meta.get("template_catalog"),
        "sample_stats": split.meta.get("sample_stats"),
        "n_train": split.meta.get("n_train"),
        "api_k": split.meta.get("api_k"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    level = split.meta.get("level", "l1")
    api_k = int(split.meta.get("api_k", 3))
    vocab = build_vocab(level=level, api_k=api_k)
    (out_dir / "vocab.json").write_text(
        json.dumps({"id_to_token": vocab.id_to_token}, indent=2),
        encoding="utf-8",
    )


def load_split(path: Path) -> list[list[str]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["tokens"])
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Phase-1 DSL datasets")
    p.add_argument("--level", choices=["l0", "l1"], default="l1")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--api-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hard", action="store_true", help="Hard L1 (k=6 ordered hold-out)")
    p.add_argument("--arm", choices=["hard", "mixture", "soft"], default="hard")
    p.add_argument("--gate-knobs", action="store_true", help="Use historical gate knobs (not freeze)")
    p.add_argument("--no-templates", action="store_true")
    p.add_argument("--out", type=str, default="results/data/l1_k3_n1000")
    args = p.parse_args(argv)

    hard = args.hard or args.arm in ("hard", "mixture")
    split = generate_dataset(
        level=args.level,
        n_train=args.n_train,
        api_k=args.api_k,
        seed=args.seed,
        hard=hard,
        arm=args.arm,
        use_templates=not args.no_templates,
        gate_knobs=args.gate_knobs,
    )
    out = Path(args.out)
    save_dataset(split, out)
    print(
        json.dumps(
            {
                "out": str(out),
                "arm": split.meta.get("arm"),
                "holdout_pair_hash": split.meta.get("holdout_pair_hash"),
                "reject_rate_train": split.meta["sample_stats"]["train"]["reject_rate"],
                "n_holdout": len(split.test_holdout),
            }
        )
    )


if __name__ == "__main__":
    main()
