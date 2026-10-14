"""NTP training for Phase-1 mini-DSL."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.generate import generate_dataset, save_dataset, load_split
from src.dsl.tokens import PAD, BOS, EOS, build_vocab, Vocab
from src.model.transformer import MiniGPT, ModelConfig, count_parameters
from src.eval import evaluate_model


class TokenDataset(Dataset):
    def __init__(self, sequences: list[list[str]], vocab: Vocab, block_size: int):
        self.vocab = vocab
        self.block_size = block_size
        self.pad_id = vocab.id(PAD)
        self.data = [vocab.encode(seq) for seq in sequences]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        ids = self.data[idx][: self.block_size]
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
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--out", type=str, default="results/runs/l1_k3_n1000")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--max-eval", type=int, default=100)
    p.add_argument(
        "--hard",
        action="store_true",
        help="Hard L1 freeze: k=6, cross-stmt deps, ordered A∘B hold-out",
    )
    p.add_argument(
        "--arm",
        choices=["hard", "mixture"],
        default="hard",
        help="Data arm: hard-only k=6 or mixture k'∈{2,3,6}",
    )
    p.add_argument(
        "--gate-knobs",
        action="store_true",
        help="Use historical difficulty-gate knobs (not freeze defaults)",
    )
    args = p.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if args.data_dir:
        data_dir = Path(args.data_dir)
        train_seqs = load_split(data_dir / "train.jsonl")
        meta = json.loads((data_dir / "meta.json").read_text())
        api_k = int(meta.get("api_k", args.api_k))
        level = meta.get("level", args.level)
    else:
        hard = args.hard or args.arm in ("hard", "mixture")
        api_k = 6 if hard else args.api_k
        print(
            f"[train] generating data n_train={args.n_train} arm={args.arm} "
            f"hard={hard} gate_knobs={args.gate_knobs} seed={args.seed}",
            flush=True,
        )
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

    vocab = build_vocab(level=level, api_k=api_k)
    print(
        f"[train] data ready n_train={len(train_seqs)} level={level} api_k={api_k} "
        f"out={out_dir}",
        flush=True,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        print(
            f"[train] cuda.is_available={torch.cuda.is_available()} "
            f"device_count={torch.cuda.device_count()} "
            f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError(
                "requested --device cuda but no CUDA device is visible; "
                "check COMPUTE GPU count and CUDA_VISIBLE_DEVICES"
            )

    cfg = ModelConfig(
        vocab_size=len(vocab),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
    )
    print(f"[train] building MiniGPT then .to({device})", flush=True)
    model = MiniGPT(cfg).to(device)
    n_params = count_parameters(model)
    print(f"vocab={len(vocab)} params={n_params:,} device={device}", flush=True)

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
        print(f"epoch {epoch:3d}  loss={avg_loss:.4f}", flush=True)

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
            "n_train": args.n_train,
            "seed": args.seed,
            "n_params": n_params,
        },
    }
    torch.save(ckpt, out_dir / "checkpoint.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "run_meta.json").write_text(
        json.dumps({**ckpt["meta"], "epochs": args.epochs, "out": str(out_dir)}, indent=2),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
