"""NTP training for Phase-1 mini-DSL."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.generate import (
    generate_dataset,
    save_dataset,
    load_split,
    copy_train_split,
)
from src.dsl.tokens import PAD, BOS, EOS, build_vocab, Vocab
from src.model.transformer import MiniGPT, ModelConfig, count_parameters
from src.eval import EVALUATION_VERSION, FREE_GEN_PREVIEW_N, evaluate_model, free_gen_latest_meta


class TokenDataset(Dataset):
    def __init__(self, sequences: list[list[str]], vocab: Vocab, block_size: int):
        self.vocab = vocab
        self.block_size = block_size
        self.pad_id = vocab.id(PAD)
        self.data = [vocab.encode(seq) for seq in sequences]
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        longest = max((len(ids) for ids in self.data), default=0)
        if longest > block_size:
            raise ValueError(
                f"Training sequence length {longest} exceeds block_size={block_size}; "
                "increase --block-size. Programs must not be silently truncated."
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        ids = self.data[idx]
        # input = all but last; target = all but first
        x = ids[:-1]
        y = ids[1:]
        # pad
        pad_len = self.block_size - 1 - len(x)
        if pad_len > 0:
            x = x + [self.pad_id] * pad_len
            y = y + [-100] * pad_len
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train MiniGPT NTP on L0/L1 DSL")
    p.add_argument("--level", choices=["l0", "l1"], default="l1")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--api-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--out", type=str, default="results/runs/l1_k3_n1000")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--max-eval", type=int, default=100)
    p.add_argument(
        "--dump-free-gen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write every free-gen program to free_gen_<split>.jsonl under --out "
            "(default: on). history.json keeps a 5-example preview plus "
            "examples_path. Pass --no-dump-free-gen to disable."
        ),
    )
    p.add_argument(
        "--hard",
        action="store_true",
        help="Hard L1 freeze: k=6, cross-stmt deps, ordered A∘B hold-out",
    )
    p.add_argument(
        "--arm",
        choices=["soft", "hard", "mixture"],
        default=None,
        help="Data arm: soft (default, honors api-k), hard k=6, or mixture k in {2,3,6}",
    )
    p.add_argument(
        "--gate-knobs",
        action="store_true",
        help="Use historical difficulty-gate knobs (not freeze defaults)",
    )
    args = p.parse_args(argv)
    if args.epochs < 1 or args.eval_every < 1 or args.max_eval < 1:
        p.error("epochs, eval-every, and max-eval must be positive")
    if args.hard and args.arm == "soft":
        p.error("--hard conflicts with --arm soft")
    args.arm = args.arm or ("hard" if args.hard else "soft")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if args.data_dir:
        data_dir = Path(args.data_dir)
        train_seqs = load_split(data_dir / "train.jsonl")
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        api_k = int(meta.get("api_k", args.api_k))
        level = meta.get("level", args.level)
    else:
        hard = args.hard or args.arm in ("hard", "mixture")
        api_k = 6 if hard else args.api_k
        split = generate_dataset(
            level=args.level,
            n_train=args.n_train,
            api_k=api_k,
            seed=args.seed,
            hard=hard,
            arm=args.arm,
            gate_knobs=args.gate_knobs,
        )
        data_dir = out_dir / "data"
        save_dataset(split, data_dir)
        train_seqs = split.train
        api_k = int(split.meta.get("api_k", api_k))
        level = args.level

    # Full train split lives in the run folder even if --data-dir is external.
    train_copy = copy_train_split(data_dir / "train.jsonl", out_dir)

    vocab = build_vocab(level=level, api_k=api_k)
    device = torch.device(args.device)

    cfg = ModelConfig(
        vocab_size=len(vocab),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
    )
    model = MiniGPT(cfg).to(device)
    n_params = count_parameters(model)
    print(f"vocab={len(vocab)} params={n_params:,} device={device}")

    ds = TokenDataset(train_seqs, vocab, args.block_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        row: dict = {"epoch": epoch, "loss": avg_loss, "sec": time.time() - t0}
        print(f"epoch {epoch:3d}  loss={avg_loss:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate_model(
                model,
                vocab,
                data_dir,
                level=level,
                api_k=api_k,
                device=device,
                max_samples=args.max_eval,
                block_size=args.block_size,
                dump_dir=out_dir if args.dump_free_gen else None,
                seed=args.seed,
                epoch=epoch,
            )
            row["metrics"] = metrics
            print(json.dumps(metrics, indent=2))

        history.append(row)

    ckpt = {
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "vocab": vocab.id_to_token,
        "meta": {
            "level": level,
            "api_k": api_k,
            "n_train": len(train_seqs),
            "seed": args.seed,
            "n_params": n_params,
            "epochs": args.epochs,
            "evaluation_version": EVALUATION_VERSION,
        },
    }
    torch.save(ckpt, out_dir / "checkpoint.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    free_gen = free_gen_latest_meta(out_dir) if args.dump_free_gen else {}
    run_meta = {
        **ckpt["meta"],
        "epochs": args.epochs,
        "evaluation_version": EVALUATION_VERSION,
        "training_args": vars(args),
        "out": str(out_dir),
        "data_dir": str(data_dir),
        **train_copy,
        "dump_free_gen": bool(args.dump_free_gen),
        "free_gen_preview": FREE_GEN_PREVIEW_N,
        "free_gen": free_gen,
        "free_gen_note": (
            "free_gen_<split>.jsonl is the latest exec-accept dump (every sample). "
            "history.json metrics.<split>.examples is a "
            f"{FREE_GEN_PREVIEW_N}-row preview; examples_path is "
            "free_gen_<split>_epoch<E>.jsonl for that eval."
        ),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
