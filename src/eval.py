"""Evaluation metrics for Phase-1 DSL NTP models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.generate import load_split
from src.dsl.continuations import legal_continuations
from src.dsl.l0 import evaluate_l0
from src.dsl.l1 import evaluate_l1
from src.dsl.tokens import BOS, EOS, PAD, Vocab, build_vocab
from src.model.transformer import MiniGPT, ModelConfig


def _strip_special(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in (BOS, EOS, PAD)]


def teacher_forced_metrics(
    model: MiniGPT,
    vocab: Vocab,
    sequences: list[list[str]],
    level: str,
    api_k: int,
    device: torch.device,
    max_samples: int,
) -> dict:
    """Legal-next-token rate and forced-unique accuracy under teacher forcing."""
    model.eval()
    eos_id = vocab.id(EOS)
    n_pos = 0
    n_legal = 0
    n_unique = 0
    n_unique_correct = 0

    with torch.no_grad():
        for seq in sequences[:max_samples]:
            ids = vocab.encode(seq)
            if len(ids) < 2:
                continue
            x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            preds = logits[0]  # T, V
            for t in range(len(ids) - 1):
                legal = legal_continuations(list(seq[: t + 1]), level=level, api_k=api_k)
                if not legal:
                    continue
                n_pos += 1
                pred_id = int(preds[t].argmax().item())
                pred_tok = vocab.id_to_token[pred_id]
                if pred_tok in legal:
                    n_legal += 1
                if len(legal) == 1:
                    n_unique += 1
                    only = next(iter(legal))
                    if pred_tok == only:
                        n_unique_correct += 1

    return {
        "n_positions": n_pos,
        "legal_next_rate": n_legal / n_pos if n_pos else 0.0,
        "forced_unique_acc": n_unique_correct / n_unique if n_unique else 0.0,
        "n_unique_positions": n_unique,
    }


@torch.no_grad()
def execution_accept_rate(
    model: MiniGPT,
    vocab: Vocab,
    sequences: list[list[str]],
    level: str,
    api_k: int,
    device: torch.device,
    max_samples: int,
    block_size: int,
    max_new_tokens: int = 48,
    temperature: float = 1.0,
) -> dict:
    """Autoregressive generation from BOS; interpreter accept rate.

    Uses temperature sampling (not greedy) so mode-collapse does not inflate
    accept rate by repeating one valid program.
    """
    model.eval()
    bos_id = vocab.id(BOS)
    eos_id = vocab.id(EOS)
    n_ok = 0
    n = 0
    examples: list[dict] = []
    unique: set[tuple[str, ...]] = set()

    for _ in sequences[:max_samples]:
        n += 1
        idx = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        for _step in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[0, -1] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            idx = torch.cat(
                [idx, torch.tensor([[next_id]], device=device)], dim=1
            )
            if next_id == eos_id:
                break
        gen_ids = idx[0].tolist()
        gen_toks = vocab.decode(gen_ids)
        body = _strip_special(gen_toks)
        unique.add(tuple(body))
        if level == "l0":
            result = evaluate_l0(body)
        else:
            result = evaluate_l1(body, api_k=api_k)
        if result.ok:
            n_ok += 1
        if len(examples) < 5:
            examples.append(
                {
                    "tokens": gen_toks,
                    "ok": result.ok,
                    "error": None if result.ok else getattr(result, "error", None),
                }
            )

    return {
        "n": n,
        "exec_accept_rate": n_ok / n if n else 0.0,
        "n_unique_programs": len(unique),
        "examples": examples,
    }


def evaluate_model(
    model: MiniGPT,
    vocab: Vocab,
    data_dir: Path | str,
    level: str = "l1",
    api_k: int = 3,
    device: torch.device | None = None,
    max_samples: int = 100,
    block_size: int = 64,
) -> dict:
    device = device or torch.device("cpu")
    data_dir = Path(data_dir)
    out: dict = {}

    for split_name in ("test_iid", "test_holdout"):
        path = data_dir / f"{split_name}.jsonl"
        if not path.exists():
            continue
        seqs = load_split(path)
        if not seqs:
            continue
        tf = teacher_forced_metrics(
            model, vocab, seqs, level, api_k, device, max_samples
        )
        ex = execution_accept_rate(
            model, vocab, seqs, level, api_k, device, max_samples, block_size
        )
        out[split_name] = {**tf, **{k: v for k, v in ex.items() if k != "examples"}, "examples": ex["examples"]}

    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-samples", type=int, default=100)
    args = p.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    vocab = Vocab(
        token_to_id={t: i for i, t in enumerate(ckpt["vocab"])},
        id_to_token=ckpt["vocab"],
    )
    cfg = ModelConfig(**ckpt["config"])
    model = MiniGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    meta = ckpt.get("meta", {})
    metrics = evaluate_model(
        model,
        vocab,
        args.data_dir,
        level=meta.get("level", "l1"),
        api_k=int(meta.get("api_k", 3)),
        device=torch.device(args.device),
        max_samples=args.max_samples,
        block_size=cfg.block_size,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
