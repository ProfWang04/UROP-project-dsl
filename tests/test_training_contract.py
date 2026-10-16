"""Regression checks for context sizing, conditioning, and RNG isolation."""

import torch
import pytest

from src.dsl.tokens import BOS, EOS, build_vocab
from src.train import TokenDataset, main as train_main
from src.eval import execution_accept_rate, teacher_forced_metrics
from src.model.transformer import MiniGPT, ModelConfig


def test_training_never_truncates_programs():
    vocab = build_vocab("l0")
    with pytest.raises(ValueError, match="silently truncated"):
        TokenDataset([[BOS, "1", "+", "2", EOS]], vocab, 4)
    ds = TokenDataset([[BOS, "1", EOS]], vocab, 3)
    assert ds[0][1].tolist() == [vocab.id("1"), vocab.id(EOS)]


def test_evaluation_context_error_is_actionable():
    vocab = build_vocab("l0")
    model = MiniGPT(ModelConfig(len(vocab), n_layer=1, n_head=1, n_embd=8, block_size=2))
    with pytest.raises(ValueError, match="increase --block-size"):
        teacher_forced_metrics(model, vocab, [[BOS, "1", "+", "2", EOS]],
                               "l0", 3, torch.device("cpu"), 1)


def test_prefix_is_used_and_sampling_preserves_global_rng():
    vocab = build_vocab("l1", 6)
    model = MiniGPT(ModelConfig(len(vocab), n_layer=1, n_head=1, n_embd=8, block_size=128))
    seq = [BOS, "set", "b", "9", "print", "b", EOS]
    rng = torch.random.get_rng_state().clone()
    kwargs = dict(level="l1", api_k=6, device=torch.device("cpu"), max_samples=1,
                  block_size=128, max_new_tokens=2, condition_on_prefix=True, seed=5)
    first = execution_accept_rate(model, vocab, [seq], **kwargs)
    second = execution_accept_rate(model, vocab, [seq], **kwargs)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert first["examples"] == second["examples"]
    assert first["examples"][0]["tokens"][:4] == seq[:4]


def test_default_generation_can_complete_more_than_48_tokens(monkeypatch):
    vocab = build_vocab("l0")
    model = MiniGPT(ModelConfig(len(vocab), n_layer=1, n_head=1, n_embd=8, block_size=128))
    target = [BOS] + ["1", "+"] * 25 + ["1", EOS]

    def forward(idx, targets=None):
        logits = torch.full((1, idx.size(1), len(vocab)), -float("inf"))
        logits[0, -1, vocab.id(target[idx.size(1)])] = 0
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    result = execution_accept_rate(model, vocab, [target], "l0", 3,
                                   torch.device("cpu"), 1, 128)
    assert result["exec_accept_rate"] == 1.0
    assert result["examples"][0]["tokens"] == target


def test_fresh_l0_training_cli(tmp_path):
    train_main(["--level", "l0", "--n-train", "4", "--epochs", "1",
                "--n-layer", "1", "--n-head", "1", "--n-embd", "8",
                "--max-eval", "1", "--out", str(tmp_path)])
    assert (tmp_path / "checkpoint.pt").is_file()
