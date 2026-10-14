"""
Two-arm N(k=6) phase curve: hard-only vs mixture/curriculum.

Primary: IID + ordered add∘mul hold-out exec-accept.
Secondary: legal-next.
Reports a coarse N*(6) boundary (smallest N with IID exec-accept ≥ threshold).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.generate import generate_l1_dataset, save_dataset, load_split
from src.dsl.tokens import PAD, build_vocab, Vocab
from src.model.transformer import MiniGPT, ModelConfig, count_parameters
from src.eval import evaluate_model
from src.train import TokenDataset


def _train_one(
    *,
    arm: str,
    n_train: int,
    seed: int,
    epochs: int,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    block_size: int,
    max_eval: int,
    lr: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    split = generate_l1_dataset(
        n_train=n_train,
        n_test_iid=min(100, max(40, n_train // 10)),
        n_test_holdout=min(100, max(40, n_train // 10)),
        seed=seed,
        hard=True,
        arm=arm,
        use_templates=True,
    )
    data_dir = out_dir / "data"
    save_dataset(split, data_dir)

    api_k = int(split.meta["api_k"])
    vocab = build_vocab(level="l1", api_k=api_k)
    cfg = ModelConfig(
        vocab_size=len(vocab),
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=block_size,
    )
    model = MiniGPT(cfg).to(device)
    ds = TokenDataset(split.train, vocab, block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_b = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            n_b += 1
        history.append({"epoch": epoch, "loss": total / max(n_b, 1)})

    metrics = evaluate_model(
        model,
        vocab,
        data_dir,
        level="l1",
        api_k=api_k,
        device=device,
        max_samples=max_eval,
        block_size=block_size,
    )
    result = {
        "arm": arm,
        "n_train": n_train,
        "seed": seed,
        "epochs": epochs,
        "n_params": count_parameters(model),
        "sec": round(time.time() - t0, 2),
        "freeze_defaults": split.meta["freeze_defaults"],
        "experiment_knobs": split.meta["experiment_knobs"],
        "holdout_pair": split.meta["holdout_pair"],
        "holdout_pair_hash": split.meta["holdout_pair_hash"],
        "sample_stats": split.meta["sample_stats"],
        "metrics": {
            "test_iid": {
                k: metrics.get("test_iid", {}).get(k)
                for k in ("legal_next_rate", "exec_accept_rate", "n", "n_unique_programs")
            },
            "test_holdout": {
                k: metrics.get("test_holdout", {}).get(k)
                for k in ("legal_next_rate", "exec_accept_rate", "n", "n_unique_programs")
            },
        },
        "final_loss": history[-1]["loss"] if history else None,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "vocab": vocab.id_to_token,
            "meta": result,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def estimate_n_star(points: list[dict], threshold: float = 0.9) -> int | None:
    """Smallest N with IID exec-accept ≥ threshold; None if never crossed."""
    ordered = sorted(points, key=lambda r: r["n_train"])
    for r in ordered:
        iid = (r.get("metrics") or {}).get("test_iid") or {}
        rate = iid.get("exec_accept_rate")
        if rate is not None and rate >= threshold:
            return int(r["n_train"])
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Two-arm N(k=6) phase curve")
    p.add_argument("--arms", nargs="+", default=["hard", "mixture"])
    p.add_argument("--n-grid", nargs="+", type=int, default=[200, 500, 1000])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-eval", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--threshold", type=float, default=0.9, help="N* IID exec-accept threshold")
    p.add_argument("--out", type=str, default="results/n_curve_k6")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for arm in args.arms:
        arm_points: list[dict] = []
        for n in args.n_grid:
            run_dir = root / f"{arm}_n{n}_s{args.seed}"
            print(f"=== arm={arm} N={n} ===")
            result = _train_one(
                arm=arm,
                n_train=n,
                seed=args.seed,
                epochs=args.epochs,
                out_dir=run_dir,
                device=device,
                batch_size=args.batch_size,
                block_size=args.block_size,
                max_eval=args.max_eval,
                lr=args.lr,
            )
            arm_points.append(result)
            all_results.append(result)
            iid = result["metrics"]["test_iid"]["exec_accept_rate"]
            ho = result["metrics"]["test_holdout"]["exec_accept_rate"]
            print(f"  IID exec={iid:.3f}  holdout exec={ho:.3f}")

        n_star = estimate_n_star(arm_points, threshold=args.threshold)
        summary = {
            "arm": arm,
            "n_grid": args.n_grid,
            "threshold": args.threshold,
            "N_star_6": n_star,
            "points": [
                {
                    "n": r["n_train"],
                    "iid_exec": r["metrics"]["test_iid"]["exec_accept_rate"],
                    "holdout_exec": r["metrics"]["test_holdout"]["exec_accept_rate"],
                    "iid_legal": r["metrics"]["test_iid"]["legal_next_rate"],
                    "holdout_legal": r["metrics"]["test_holdout"]["legal_next_rate"],
                }
                for r in arm_points
            ],
        }
        (root / f"summary_{arm}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"arm={arm} N*(6) @≥{args.threshold}: {n_star}")

    curve_manifest = {
        "description": "Two-arm N(k=6) curve with Cadmus-style template sampling",
        "freeze_defaults": all_results[0]["freeze_defaults"] if all_results else {},
        "gate_knobs_reference_note": "Not used unless --gate-knobs at data gen; curve uses freeze defaults",
        "holdout_pair_hash": all_results[0]["holdout_pair_hash"] if all_results else None,
        "arms": args.arms,
        "n_grid": args.n_grid,
        "seed": args.seed,
        "epochs": args.epochs,
        "threshold": args.threshold,
        "results": all_results,
        "N_star_by_arm": {
            arm: estimate_n_star([r for r in all_results if r["arm"] == arm], args.threshold)
            for arm in args.arms
        },
    }
    (root / "curve_manifest.json").write_text(
        json.dumps(curve_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(curve_manifest["N_star_by_arm"], indent=2))


if __name__ == "__main__":
    main()
