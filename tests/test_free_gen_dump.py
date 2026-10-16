"""Full free-gen dumps and train-split copies.

history.json keeps a short ``examples`` preview. The jsonl on disk has every sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data.generate import (
    copy_train_split,
    generate_l0_dataset,
    save_dataset,
)
from src.dsl.tokens import BOS, EOS, PAD, build_vocab
from src.eval import (
    FREE_GEN_PREVIEW_N,
    evaluate_model,
    execution_accept_rate,
    free_gen_latest_meta,
    free_gen_paths,
)
from src.model.transformer import MiniGPT, ModelConfig
from src.train import main as train_main


def _tiny_model(vocab) -> MiniGPT:
    cfg = ModelConfig(
        vocab_size=len(vocab),
        n_layer=1,
        n_head=4,
        n_embd=32,
        block_size=32,
        dropout=0.0,
    )
    model = MiniGPT(cfg)
    model.eval()
    return model


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_free_gen_dump_writes_every_sample(tmp_path: Path):
    n = 7
    vocab = build_vocab(level="l0")
    model = _tiny_model(vocab)
    latest, epoch_path = free_gen_paths(tmp_path, "test_iid", epoch=3)
    assert epoch_path is not None

    torch.manual_seed(0)
    result = execution_accept_rate(
        model,
        vocab,
        [[BOS, EOS]] * n,
        level="l0",
        api_k=3,
        device=torch.device("cpu"),
        max_samples=n,
        block_size=32,
        max_new_tokens=3,
        dump_paths=[latest, epoch_path],
        examples_path=epoch_path,
        seed=11,
        epoch=3,
        split="test_iid",
    )

    rows = _rows(latest)
    assert len(rows) == n
    assert _rows(epoch_path) == rows
    assert result["n"] == n
    assert len(result["examples"]) == FREE_GEN_PREVIEW_N
    assert result["examples_path"] == str(epoch_path)
    assert set(result["examples"][0]) == {"tokens", "ok", "error"}
    assert result["examples"][0]["tokens"] == rows[0]["tokens"]

    n_ok = sum(1 for row in rows if row["ok"])
    assert result["exec_accept_rate"] == n_ok / n
    bodies = {
        tuple(row["tokens"][1:-1] if row["tokens"][-1] == EOS else row["tokens"][1:]) for row in rows
    }
    assert result["n_unique_programs"] == len(bodies)
    assert [row["step"] for row in rows] == list(range(n))
    assert all(row["seed"] == 11 and row["epoch"] == 3 and row["split"] == "test_iid" for row in rows)
    assert all(isinstance(row["n_new_tokens"], int) and row["n_new_tokens"] >= 1 for row in rows)

    # A later eval overwrites the latest file and leaves the earlier epoch dump intact.
    epoch_text = epoch_path.read_text(encoding="utf-8")
    latest2, epoch2 = free_gen_paths(tmp_path, "test_iid", epoch=4)
    assert latest2 == latest
    torch.manual_seed(1)
    execution_accept_rate(
        model,
        vocab,
        [[BOS, EOS]] * n,
        level="l0",
        api_k=3,
        device=torch.device("cpu"),
        max_samples=n,
        block_size=32,
        max_new_tokens=3,
        dump_paths=[latest2, epoch2],
        examples_path=epoch2,
        seed=11,
        epoch=4,
        split="test_iid",
    )
    assert epoch_path.read_text(encoding="utf-8") == epoch_text
    assert latest.read_text(encoding="utf-8") == epoch2.read_text(encoding="utf-8")
    assert len(_rows(latest)) == n


def test_accept_rate_uses_full_sample_not_preview(tmp_path: Path, monkeypatch):
    """First 5 programs fail and the rest pass; the rate must count all of them."""
    calls = {"n": 0}

    def fake_eval(_body):
        calls["n"] += 1
        ok = calls["n"] > FREE_GEN_PREVIEW_N
        return type("R", (), {"ok": ok, "error": None if ok else "nope"})()

    monkeypatch.setattr("src.dsl.verification.evaluate_l0", fake_eval)
    # Deterministic complete programs keep this test about full-sample counting.
    def scripted_forward(self, idx, targets=None):
        logits = torch.full((1, idx.size(1), self.cfg.vocab_size), -float("inf"))
        logits[:, -1, vocab.id("1" if idx.size(1) == 1 else EOS)] = 0
        return logits, None
    monkeypatch.setattr(MiniGPT, "forward", scripted_forward)
    n = 8
    vocab = build_vocab(level="l0")
    model = _tiny_model(vocab)
    path = tmp_path / "free_gen_test_iid.jsonl"
    torch.manual_seed(0)
    result = execution_accept_rate(
        model,
        vocab,
        [[BOS, EOS]] * n,
        level="l0",
        api_k=3,
        device=torch.device("cpu"),
        max_samples=n,
        block_size=32,
        max_new_tokens=2,
        dump_paths=[path],
        examples_path=path,
        seed=0,
        split="test_iid",
    )
    rows = _rows(path)
    assert len(rows) == n
    assert result["n"] == n
    assert len(result["examples"]) == FREE_GEN_PREVIEW_N
    assert all(not ex["ok"] for ex in result["examples"])
    assert result["exec_accept_rate"] == (n - FREE_GEN_PREVIEW_N) / n
    assert sum(1 for row in rows if row["ok"]) == n - FREE_GEN_PREVIEW_N


def test_preview_only_when_dump_disabled():
    vocab = build_vocab(level="l0")
    model = _tiny_model(vocab)
    torch.manual_seed(0)
    result = execution_accept_rate(
        model,
        vocab,
        [[BOS, EOS]] * 6,
        level="l0",
        api_k=3,
        device=torch.device("cpu"),
        max_samples=6,
        block_size=32,
        max_new_tokens=1,
    )
    assert result["n"] == 6
    assert len(result["examples"]) == FREE_GEN_PREVIEW_N
    assert "examples_path" not in result
    assert "exec_accept_rate" in result
    assert "n_unique_programs" in result


def test_evaluate_model_names_split_dumps(tmp_path: Path):
    n_test = 8
    split = generate_l0_dataset(n_train=4, n_test=n_test, seed=0)
    data_dir = tmp_path / "data"
    save_dataset(split, data_dir)
    vocab = build_vocab(level="l0")
    model = _tiny_model(vocab)
    torch.manual_seed(0)
    metrics = evaluate_model(
        model,
        vocab,
        data_dir,
        level="l0",
        device=torch.device("cpu"),
        max_samples=n_test,
        block_size=32,
        dump_dir=tmp_path,
        seed=2,
        epoch=1,
    )
    latest = tmp_path / "free_gen_test_iid.jsonl"
    epoch_path = tmp_path / "free_gen_test_iid_epoch1.jsonl"
    assert latest.is_file() and epoch_path.is_file()
    assert len(_rows(latest)) == n_test
    assert _rows(epoch_path) == _rows(latest)
    iid = metrics["test_iid"]
    assert iid["examples_path"] == str(epoch_path)
    assert len(iid["examples"]) == FREE_GEN_PREVIEW_N
    assert iid["n"] == n_test
    assert "legal_next_rate" in iid
    assert "test_holdout" not in metrics
    assert not (tmp_path / "free_gen_test_holdout.jsonl").exists()
    meta = free_gen_latest_meta(tmp_path)
    assert meta["test_iid"]["n"] == n_test
    assert meta["test_iid"]["path"] == str(latest)


def test_train_run_copies_train_and_dumps_free_gen(tmp_path: Path):
    """src.train writes a self-contained run dir (the path job.sh calls)."""
    n_train = 20
    n_test = 30
    max_eval = 9
    data_dir = tmp_path / "data"
    save_dataset(generate_l0_dataset(n_train=n_train, n_test=n_test, seed=1), data_dir)
    out = tmp_path / "run"
    train_main(
        [
            "--level",
            "l0",
            "--data-dir",
            str(data_dir),
            "--n-train",
            str(n_train),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--eval-every",
            "1",
            "--max-eval",
            str(max_eval),
            "--n-layer",
            "1",
            "--n-embd",
            "32",
            "--n-head",
            "4",
            "--block-size",
            "32",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--dump-free-gen",
            "--out",
            str(out),
        ]
    )
    train_rows = _rows(out / "train.jsonl")
    assert len(train_rows) == n_train
    assert (out / "train.jsonl").read_text(encoding="utf-8") == (data_dir / "train.jsonl").read_text(
        encoding="utf-8"
    )
    iid_rows = _rows(out / "free_gen_test_iid.jsonl")
    assert len(iid_rows) == max_eval
    assert len(_rows(out / "free_gen_test_iid_epoch1.jsonl")) == max_eval
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    iid = history[-1]["metrics"]["test_iid"]
    assert iid["n"] == max_eval
    assert len(iid["examples"]) == FREE_GEN_PREVIEW_N
    assert set(iid["examples"][0]) == {"tokens", "ok", "error"}
    assert iid["examples_path"] == str(out / "free_gen_test_iid_epoch1.jsonl")
    assert "legal_next_rate" in iid and "exec_accept_rate" in iid and "n_unique_programs" in iid
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["n_train_saved"] == n_train
    assert meta["train_jsonl"] == str(out / "train.jsonl")
    assert meta["train_jsonl_source"] == str(data_dir / "train.jsonl")
    assert meta["data_dir"] == str(data_dir)
    assert meta["dump_free_gen"] is True
    assert meta["free_gen"]["test_iid"]["n"] == max_eval
    assert meta["free_gen_preview"] == FREE_GEN_PREVIEW_N


def test_copy_train_split_keeps_every_program(tmp_path: Path):
    source_dir = tmp_path / "external"
    source_dir.mkdir()
    source = source_dir / "train.jsonl"
    n = 12
    source.write_text(
        "".join(json.dumps({"tokens": ["<bos>", str(i), "<eos>"]}) + "\n" for i in range(n)),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    info = copy_train_split(source, run_dir)
    dest = run_dir / "train.jsonl"
    assert Path(info["train_jsonl"]) == dest
    assert info["train_jsonl_source"] == str(source)
    assert info["n_train_saved"] == n
    assert dest.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert len(_rows(dest)) == n

    # Copying onto itself must not truncate the file.
    again = copy_train_split(dest, run_dir)
    assert again["n_train_saved"] == n
    assert len(_rows(dest)) == n
