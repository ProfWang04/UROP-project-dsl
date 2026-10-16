"""Generate L0/L1 datasets with IID and composition hold-out splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
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
    accept_hard_program,
    force_unique_hard_program,
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
    progress_every = max(100, n_train // 20)

    def _progress(split_name: str, have: int, need: int, dups: int) -> None:
        print(
            f"[generate] {split_name}: {have}/{need} unique "
            f"(dup_streak={dups}, arm={arm}, gate_knobs={gate_knobs})",
            flush=True,
        )

    def sample_candidate(
        stats: SampleStats,
        *,
        holdout: bool,
        salt: int,
        prefer_procedural: bool = False,
    ) -> tuple[list[str], str] | None:
        if arm == "mixture" and not holdout:
            # Train mixture; IID hard path uses holdout=False templates below
            toks, tid, _k = sample_mixture_curriculum(
                rng, knobs=knobs, min_deps=deps, stats=stats
            )
            return toks, tid

        if use_hard and (arm in ("hard", "mixture") or hard):
            if use_templates:
                toks, tid = sample_with_templates(
                    rng,
                    holdout=holdout,
                    cfg=cfg,
                    knobs=knobs,
                    min_deps=deps,
                    min_distinct_apis=knobs.min_distinct_apis,
                    holdout_pair=(api_a, api_b),
                    stats=stats,
                    salt=salt,
                    prefer_procedural=prefer_procedural,
                )
                ok, _ = accept_hard_program(
                    toks,
                    apis=apis,
                    min_deps=deps,
                    min_distinct_apis=knobs.min_distinct_apis,
                    require_ordered_holdout=True if holdout else None,
                    forbid_ordered_holdout=(not holdout),
                    holdout_pair=(api_a, api_b),
                )
                return (toks, tid) if ok else None
            from src.dsl.hard_l1 import generate_hard_l1, generate_hard_holdout_combo

            if holdout:
                toks = generate_hard_holdout_combo(
                    cfg, api_a, api_b, rng, min_deps=deps,
                    min_distinct_apis=knobs.min_distinct_apis, salt=salt,
                )
            else:
                toks = generate_hard_l1(
                    cfg, rng, min_deps=deps,
                    min_distinct_apis=knobs.min_distinct_apis,
                    forbid_ordered_holdout=True, holdout_pair=(api_a, api_b),
                    salt=salt,
                )
            ok, _ = accept_hard_program(
                toks, apis=apis, min_deps=deps,
                min_distinct_apis=knobs.min_distinct_apis,
                require_ordered_holdout=True if holdout else None,
                forbid_ordered_holdout=(not holdout),
                holdout_pair=(api_a, api_b),
            )
            return (toks, "procedural") if ok else None

        # soft toy path
        if holdout:
            toks = generate_l1_holdout_combo(cfg, api_a, api_b, rng)
            if not (api_a in toks and api_b in toks):
                return None
        else:
            toks = generate_l1(cfg, rng)
            if api_a in toks and api_b in toks:
                return None
        if not evaluate_l1(toks, api_k=api_k_eval).ok:
            return None
        return toks, "soft"

    def fill_unique(
        n_need: int,
        *,
        holdout: bool,
        stats: SampleStats,
        split_name: str,
        seen: set[tuple[str, ...]],
        record_tid: bool = False,
    ) -> list[list[str]]:
        out: list[list[str]] = []
        dup_streak = 0
        attempts = 0
        max_attempts = max(50_000, n_need * 50)
        while len(out) < n_need and attempts < max_attempts:
            attempts += 1
            # Bias to procedural only after duplicate streak (hang escape).
            cand = sample_candidate(
                stats,
                holdout=holdout,
                salt=attempts + len(seen) * 7,
                prefer_procedural=(dup_streak >= 8),
            )
            if cand is None:
                dup_streak += 1
                continue
            toks, tid = cand
            key = tuple(toks)
            if key in seen:
                dup_streak += 1
                if dup_streak >= 20 and use_hard:
                    forced = force_unique_hard_program(
                        seen, cfg, rng,
                        holdout=holdout,
                        min_deps=deps,
                        min_distinct_apis=knobs.min_distinct_apis,
                        holdout_pair=(api_a, api_b),
                    )
                    if forced is None:
                        print(
                            f"[generate] WARN {split_name}: force_unique failed "
                            f"at {len(out)}/{n_need}",
                            flush=True,
                        )
                        break
                    toks, tid = forced, "force_unique"
                    key = tuple(toks)
                    dup_streak = 0
                else:
                    continue
            else:
                dup_streak = 0
            seen.add(key)
            out.append(_wrap(toks))
            if record_tid:
                template_ids_train.append(tid)
            if len(out) % progress_every == 0 or len(out) == n_need:
                _progress(split_name, len(out), n_need, dup_streak)
        if len(out) < n_need:
            raise RuntimeError(
                f"Failed to gather {n_need} unique {split_name} programs "
                f"(got {len(out)} after {attempts} attempts; gate_knobs={gate_knobs})"
            )
        return out

    print(
        f"[generate] start arm={arm} n_train={n_train} gate_knobs={gate_knobs} "
        f"knobs={knobs.to_dict()} holdout={api_a}->{api_b}",
        flush=True,
    )
    seen: set[tuple[str, ...]] = set()
    train = fill_unique(
        n_train, holdout=False, stats=train_stats, split_name="train",
        seen=seen, record_tid=True,
    )

    # IID: for mixture arm, evaluate on hard k=6 distribution
    def sample_iid_hard(stats: SampleStats, salt: int, prefer: bool):
        return sample_with_templates(
            rng, holdout=False, cfg=cfg, knobs=knobs, min_deps=deps,
            min_distinct_apis=knobs.min_distinct_apis,
            holdout_pair=(api_a, api_b), stats=stats, salt=salt,
            prefer_procedural=prefer or gate_knobs,
        )

    test_iid: list[list[str]] = []
    if arm == "mixture":
        dup_streak = 0
        attempts = 0
        while len(test_iid) < n_test_iid and attempts < max(50_000, n_test_iid * 50):
            attempts += 1
            toks, _tid = sample_iid_hard(iid_stats, attempts, prefer=dup_streak >= 8)
            ok, _ = accept_hard_program(
                toks, apis=apis, min_deps=deps,
                min_distinct_apis=knobs.min_distinct_apis,
                forbid_ordered_holdout=True, holdout_pair=(api_a, api_b),
            )
            if not ok:
                continue
            key = tuple(toks)
            if key in seen:
                dup_streak += 1
                if dup_streak >= 20:
                    forced = force_unique_hard_program(
                        seen, cfg, rng, holdout=False, min_deps=deps,
                        min_distinct_apis=knobs.min_distinct_apis,
                        holdout_pair=(api_a, api_b),
                    )
                    if forced is None:
                        break
                    toks = forced
                    key = tuple(toks)
                    dup_streak = 0
                else:
                    continue
            else:
                dup_streak = 0
            seen.add(key)
            test_iid.append(_wrap(toks))
            if len(test_iid) % max(20, n_test_iid // 5) == 0:
                _progress("test_iid", len(test_iid), n_test_iid, dup_streak)
    else:
        test_iid = fill_unique(
            n_test_iid, holdout=False, stats=iid_stats, split_name="test_iid",
            seen=seen,
        )

    if len(test_iid) != n_test_iid:
        raise RuntimeError(f"Failed to gather {n_test_iid} unique test_iid programs (got {len(test_iid)})")

    test_holdout = fill_unique(
        n_test_holdout, holdout=True, stats=holdout_stats, split_name="test_holdout",
        seen=seen,
    )
    print(
        f"[generate] done train={len(train)} iid={len(test_iid)} "
        f"holdout={len(test_holdout)} "
        f"train_reject_rate={train_stats.reject_rate:.3f}",
        flush=True,
    )

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
        for option in ("gate_knobs", "use_templates"):
            kwargs.pop(option, None)
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


def load_split(path: Path | str) -> list[list[str]]:
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["tokens"])
    return rows


def count_jsonl_lines(path: Path | str) -> int:
    """Count non-empty lines in a jsonl file."""
    path = Path(path)
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def copy_train_split(source: Path | str, dest_dir: Path | str) -> dict:
    """Copy the full train jsonl into a run directory.

    Destination is always ``<dest_dir>/train.jsonl`` so each run folder is
    self-contained even when training read an external ``--data-dir``.
    If source and dest are the same file, the copy is skipped.

    Returns fields meant for ``run_meta.json``.
    """
    source = Path(source)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "train.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"train split not found: {source}")
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    return {
        "train_jsonl": str(dest),
        "train_jsonl_source": str(source),
        "n_train_saved": count_jsonl_lines(dest),
    }


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
                "reject_rate_train": split.meta.get("sample_stats", {}).get("train", {}).get("reject_rate"),
                "n_holdout": len(split.test_holdout),
            }
        )
    )


if __name__ == "__main__":
    main()
