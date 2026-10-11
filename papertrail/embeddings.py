"""Pretrained input embeddings for Part II: loading, missing words, freezing.

Two sources, chosen by `model.emb_init` in the config:

    word2vec   the skip-gram vectors trained by papertrail/word2vec.py
    glove      the public GloVe vectors (scripts/download_glove.sh)

Either way the vectors are copied into the model's input embedding table, one
row per vocabulary token. A token the source does not have keeps a random row.
"""
from pathlib import Path

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch import nn

from papertrail.config import REPO_ROOT
from papertrail.data import Vocab


def load_vectors(path: Path, wanted: set[str]) -> dict[str, np.ndarray]:
    """Read the vectors for the tokens in `wanted` from a .kv or a GloVe .txt file."""
    if path.suffix == ".kv":
        vectors = KeyedVectors.load(str(path))
        # .copy(): gensim hands out read-only views into its own table.
        return {token: vectors[token].copy() for token in wanted if token in vectors.key_to_index}

    # GloVe text format: one token per line, followed by its numbers, all
    # separated by single spaces. The file has 400,000 lines and we need a few
    # thousand of them, so only the wanted lines are converted to numbers.
    found = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            token, _, numbers = line.rstrip("\n").partition(" ")
            if token in wanted:
                found[token] = np.array(numbers.split(" "), dtype=np.float32)
    return found


def build_table(vocab: Vocab, vectors: dict[str, np.ndarray], dim: int, seed: int) -> tuple[torch.Tensor, list[str]]:
    """Arrange the vectors into a (vocab_size, dim) table in vocabulary order.

    Missing tokens get rows drawn from a normal distribution with mean 0 and
    the same standard deviation as the vectors that were found, so they sit at
    the same scale as the rest of the table. The draw has its own seeded
    generator: it does not depend on, or disturb, the model's initialization.

    Returns the table and the list of missing tokens.
    """
    found = torch.tensor(np.stack(list(vectors.values())))
    assert found.shape[1] == dim, f"vectors have dimension {found.shape[1]}, the model expects {dim}"

    generator = torch.Generator().manual_seed(seed)
    table = torch.randn(len(vocab), dim, generator=generator) * found.std()

    missing = []
    for i, token in enumerate(vocab.itos):
        if token in vectors:
            table[i] = torch.from_numpy(vectors[token])
        else:
            missing.append(token)
    return table, missing


def init_embeddings(model: nn.Module, config: dict, vocab: Vocab, train_stream: torch.Tensor) -> dict:
    """Set up the model's input embedding table as the config asks.

    Returns a description of what was done, for the results file: the source,
    whether the table is frozen, and how much of the vocabulary it covers.
    """
    settings = config["model"]
    if settings["emb_init"] == "random":
        assert not settings["freeze_emb"], "a frozen random table is not one of the Part II settings"
        return {"init": "random", "frozen": False}

    if not hasattr(model, "embedding"):
        raise NotImplementedError("pretrained embeddings are only set up for the RNN and LSTM models")
    assert settings["oov_init"] == "normal_matched_std", f"unknown oov_init: {settings['oov_init']!r}"

    vectors = load_vectors(REPO_ROOT / settings["emb_path"], set(vocab.itos))
    table, missing = build_table(vocab, vectors, settings["emb_dim"], config["seed"])

    with torch.no_grad():
        model.embedding.weight.copy_(table)
    if settings["freeze_emb"]:
        # The whole table, including the rows of missing and special tokens.
        # A parameter that does not require gradients is never updated, and
        # train.py also leaves it out of the optimizer.
        model.embedding.weight.requires_grad_(False)

    # Coverage two ways: by vocabulary entry, and by how often those entries
    # actually occur in the training text.
    missing_ids = torch.tensor([vocab.stoi[token] for token in missing], dtype=torch.long)
    missing_occurrences = torch.isin(train_stream, missing_ids).sum().item()
    return {
        "init": settings["emb_init"],
        "frozen": settings["freeze_emb"],
        "path": settings["emb_path"],
        "oov_init": settings["oov_init"],
        "vocabulary": len(vocab),
        "covered": len(vocab) - len(missing),
        "coverage_of_vocabulary": 1 - len(missing) / len(vocab),
        "coverage_of_train_tokens": 1 - missing_occurrences / len(train_stream),
        "vector_std": table.std().item(),
        "missing_examples": missing[:30],
    }
