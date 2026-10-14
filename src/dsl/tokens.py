"""Controlled vocabulary for L0/L1 (~30–80 tokens)."""

from __future__ import annotations

from dataclasses import dataclass, field


# Special tokens
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

# L0 arithmetic
DIGITS = [str(i) for i in range(10)]
OPS = ["+", "-", "*"]
PARENS = ["(", ")"]

# L1 core keywords / APIs (k selects how many beyond set/print)
CORE_APIS = ["set", "print"]
# Extra APIs ordered; first k-2 extras are enabled when k > 2
EXTRA_APIS = [
    "inc",  # inc x  — x = x + 1
    "dec",  # dec x
    "add",  # add x y — x = x + y
    "mul",  # mul x y — x = x * y
    "clr",  # clr x  — x = 0
    "mov",  # mov x y — x = y
    "neg",  # neg x  — x = 0 - x  (for k≥9)
    "swap",  # swap x y — exchange (for k≥10)
]

# Variable identifiers (small fixed set)
VARS = list("abcdefgh")


@dataclass
class Vocab:
    """Bidirectional token ↔ id map."""

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token_to_id[UNK]
        return [self.token_to_id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] for i in ids]

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def id(self, token: str) -> int:
        return self.token_to_id[token]


def build_vocab(level: str = "l1", api_k: int = 3) -> Vocab:
    """
    Build vocab for L0 or L1.

    api_k: number of statement APIs for L1 (min 2 = set+print).
    Typical range 2–8 → total vocab ~40–50.
    """
    tokens: list[str] = [PAD, BOS, EOS, UNK]
    tokens.extend(DIGITS)
    tokens.extend(OPS)
    tokens.extend(PARENS)

    if level == "l1":
        k = max(2, min(api_k, 2 + len(EXTRA_APIS)))
        apis = CORE_APIS + EXTRA_APIS[: k - 2]
        tokens.extend(apis)
        tokens.extend(VARS)

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    token_to_id = {t: i for i, t in enumerate(ordered)}
    return Vocab(token_to_id=token_to_id, id_to_token=ordered)


def enabled_apis(api_k: int) -> list[str]:
    k = max(2, min(api_k, 2 + len(EXTRA_APIS)))
    return CORE_APIS + EXTRA_APIS[: k - 2]
