"""Evaluation metrics for Phase-1 DSL NTP models.

Free-gen (exec-accept) writes every sampled program to jsonl when a dump
directory is set. The returned ``examples`` list stays a short preview so
existing history.json consumers keep working; ``examples_path`` points at
the full dump.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.generate import count_jsonl_lines, load_split
from src.dsl.continuations import legal_continuations
from src.dsl.l0 import evaluate_l0
from src.dsl.l1 import evaluate_l1
from src.dsl.tokens import BOS, EOS, PAD, Vocab, build_vocab
from src.model.transformer import MiniGPT, ModelConfig

# history.json keeps this many free-gen rows inline. The jsonl dump is complete.
FREE_GEN_PREVIEW_N = 5


def _strip_special(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in (BOS, EOS, PAD)]


def free_gen_paths(
    dump_dir: Path | str, split: str, epoch: int | None = None
) -> tuple[Path, Path | None]:
    """Latest dump path, and the per-epoch path when ``epoch`` is set.

    ``free_gen_<split>.jsonl`` is overwritten on each eval so the run folder
    always has the newest full sample. ``free_gen_<split>_epoch<E>.jsonl``
    is stable for that eval and is what ``examples_path`` records.
    """
    dump_dir = Path(dump_dir)
    latest = dump_dir / f"free_gen_{split}.jsonl"
    epoch_path = (
        dump_dir / f"free_gen_{split}_epoch{epoch}.jsonl" if epoch is not None else None
    )
    return latest, epoch_path


def free_gen_latest_meta(
    dump_dir: Path | str,
    splits: tuple[str, ...] = ("test_iid", "test_holdout"),
) -> dict[str, dict]:
    """Path and line count for each free-gen dump that exists under ``dump_dir``."""
    out: dict[str, dict] = {}
    for split in splits:
        latest, _ = free_gen_paths(dump_dir, split, epoch=None)
        if latest.is_file():
            out[split] = {"path": str(latest), "n": count_jsonl_lines(latest)}
    return out


def _free_gen_record(
    *,
    step: int,
    tokens: list[str],
    ok: bool,
    error: str | None,
    n_new_tokens: int,
    seed: int | None,
    epoch: int | None,
    split: str | None,
) -> dict:
    rec: dict = {
        "step": step,
        "tokens": tokens,
        "ok": ok,
        "error": error,
        "n_new_tokens": n_new_tokens,
    }
    if seed is not None:
        rec["seed"] = seed
    if epoch is not None:
        rec["epoch"] = epoch
    if split is not None:
        rec["split"] = split
    return rec


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
    preview_n: int = FREE_GEN_PREVIEW_N,
    dump_paths: list[Path | str] | None = None,
    examples_path: Path | str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
    split: str | None = None,
) -> dict:
    """Autoregressive generation from BOS; interpreter accept rate.

    Uses temperature sampling (not greedy) so mode-collapse does not inflate
    accept rate by repeating one valid program.

    ``examples`` is only a preview (default 5). When ``dump_paths`` is set,
    every sample is written as one jsonl record (tokens, ok, error, step,
    and seed/epoch/split when provided). Rates still use the full sample.
    """
    model.eval()
    bos_id = vocab.id(BOS)
    eos_id = vocab.id(EOS)
    n_ok = 0
    n = 0
    examples: list[dict] = []
    unique: set[tuple[str, ...]] = set()
    paths = [Path(p) for p in (dump_paths or [])]
    handles: list = []
    advertised: str | None = None

    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handles.append(path.open("w", encoding="utf-8"))
        if examples_path is not None:
            advertised = str(examples_path)
        elif paths:
            advertised = str(paths[0])

        for step, _seq in enumerate(sequences[:max_samples]):
            n += 1
            idx = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            n_new_tokens = 0
            for _gen_step in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = model(idx_cond)
                logits = logits[0, -1] / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
                idx = torch.cat(
                    [idx, torch.tensor([[next_id]], device=device)], dim=1
                )
                n_new_tokens += 1
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
            error = None if result.ok else getattr(result, "error", None)
            if len(examples) < preview_n:
                examples.append(
                    {
                        "tokens": gen_toks,
                        "ok": result.ok,
                        "error": error,
                    }
                )
            if handles:
                record = _free_gen_record(
                    step=step,
                    tokens=gen_toks,
                    ok=result.ok,
                    error=error,
                    n_new_tokens=n_new_tokens,
                    seed=seed,
                    epoch=epoch,
                    split=split,
                )
                line = json.dumps(record) + "\n"
                for handle in handles:
                    handle.write(line)
    finally:
        for handle in handles:
            handle.close()

    out = {
        "n": n,
        "exec_accept_rate": n_ok / n if n else 0.0,
        "n_unique_programs": len(unique),
        "examples": examples,
    }
    if advertised is not None:
        out["examples_path"] = advertised
    return out


def evaluate_model(
    model: MiniGPT,
    vocab: Vocab,
    data_dir: Path | str,
    level: str = "l1",
    api_k: int = 3,
    device: torch.device | None = None,
    max_samples: int = 100,
    block_size: int = 64,
    dump_dir: Path | str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
    preview_n: int = FREE_GEN_PREVIEW_N,
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
        dump_paths: list[Path] = []
        advertised: Path | None = None
        if dump_dir is not None:
            latest, epoch_path = free_gen_paths(dump_dir, split_name, epoch)
            dump_paths = [latest] if epoch_path is None else [latest, epoch_path]
            advertised = epoch_path or latest
        ex = execution_accept_rate(
            model,
            vocab,
            seqs,
            level,
            api_k,
            device,
            max_samples,
            block_size,
            preview_n=preview_n,
            dump_paths=dump_paths,
            examples_path=advertised,
            seed=seed,
            epoch=epoch,
            split=split_name,
        )
        out[split_name] = {
            **tf,
            **{k: v for k, v in ex.items() if k != "examples"},
            "examples": ex["examples"],
        }

    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument(
        "--out",
        type=str,
        default="",
        help=(
            "Run directory for full free-gen jsonl dumps and a train.jsonl copy. "
            "Omit to print metrics only (examples stay a 5-row preview)."
        ),
    )
    p.add_argument("--seed", type=int, default=None)
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
    dump_dir: Path | None = None
    if args.out:
        from src.data.generate import copy_train_split

        dump_dir = Path(args.out)
        dump_dir.mkdir(parents=True, exist_ok=True)
        train_src = Path(args.data_dir) / "train.jsonl"
        if train_src.is_file():
            copy_train_split(train_src, dump_dir)
    metrics = evaluate_model(
        model,
        vocab,
        args.data_dir,
        level=meta.get("level", "l1"),
        api_k=int(meta.get("api_k", 3)),
        device=torch.device(args.device),
        max_samples=args.max_samples,
        block_size=cfg.block_size,
        dump_dir=dump_dir,
        seed=args.seed if args.seed is not None else meta.get("seed"),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
