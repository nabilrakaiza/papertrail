"""Extra measurements for the Part I discussion.

    python -m papertrail.analysis

Writes results/analysis_part1.json with three things:

    ngram          how sparse trigram and 4-gram counts are on this corpus, and
                   the trigram's perplexity on its own training text
    by_position    test loss at each position of a 64-token chunk, per model:
                   how much each model gains from having more context
    generations    repetition and special-token counts in the Part I samples
"""
import json
import math
from collections import Counter

import torch
from torch.nn import functional as F

from papertrail import ngram
from papertrail.config import REPO_ROOT, load_config
from papertrail.data import chunk, load_corpus
from papertrail.models import load_checkpoint
from papertrail.utils import pick_device

NEURAL = ("rnn", "lstm", "transformer")


def ngram_sparsity(train: list[int], order: int) -> dict:
    """How many distinct n-grams there are, and how many occur only once."""
    counts = Counter(zip(*(train[i:] for i in range(order))))
    return {
        "occurrences": sum(counts.values()),
        "distinct": len(counts),
        "seen_once": sum(1 for c in counts.values() if c == 1),
        "share_of_distinct_seen_once": sum(1 for c in counts.values() if c == 1) / len(counts),
    }


def ngram_by_position(model: ngram.KneserNeyTrigram, test: list[int], context_length: int) -> list[float]:
    """The trigram's test loss grouped by where each target falls in a chunk.

    The trigram always sees two tokens of history (except at the very start),
    so this should be flat; it is the reference line for the neural models.
    """
    sums, counts = [0.0] * context_length, [0] * context_length
    for i in range(1, len(test)):
        context = (test[i - 1],) if i == 1 else (test[i - 2], test[i - 1])
        position = (i - 1) % context_length
        sums[position] -= math.log(model.prob(test[i], context))
        counts[position] += 1
    return [s / c for s, c in zip(sums, counts)]


def neural_by_position(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor, pad_id: int) -> list[float]:
    """Mean test loss at each position 0..63 of a chunk.

    Position t is predicted from t + 1 tokens of history, because every chunk
    starts from a fresh state (the evaluation protocol). A model that makes
    good use of longer context should keep improving as t grows.
    """
    sums = torch.zeros(inputs.size(1))
    counts = torch.zeros(inputs.size(1))
    with torch.no_grad():
        for start in range(0, len(inputs), 32):
            x, y = inputs[start:start + 32], targets[start:start + 32]
            logits, _ = model(x)
            loss = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=pad_id, reduction="none")  # (batch, time)
            real = (y != pad_id).float()
            sums += (loss * real).sum(0).cpu()
            counts += real.sum(0).cpu()
    return (sums / counts).tolist()


def repetition(tokens: list[str]) -> dict:
    """Simple repetition measures for one generated continuation."""
    bigrams = list(zip(tokens, tokens[1:]))
    trigrams = Counter(zip(tokens, tokens[1:], tokens[2:]))
    return {
        "distinct_bigrams": len(set(bigrams)) / len(bigrams),
        "repeated_trigram_tokens": sum(c for c in trigrams.values() if c > 1) / sum(trigrams.values()),
        "unk": tokens.count("<unk>"),
        "eos": tokens.count("<eos>"),
    }


def main() -> None:
    config = load_config("configs/base.yaml")
    vocab, streams = load_corpus(config)
    train, test = streams["train"].tolist(), streams["test"].tolist()
    context_length = config["train"]["context_length"]
    out = {}

    ngram_config = load_config("configs/part1/ngram.yaml")
    trigram = ngram.KneserNeyTrigram(ngram.count_ngrams(train), ngram_config["model"]["discount"], len(vocab))
    out["ngram"] = {
        "trigrams": ngram_sparsity(train, 3),
        "fourgrams": ngram_sparsity(train, 4),
        "train_perplexity": math.exp(ngram.evaluate(trigram, train)),
        "test_perplexity": math.exp(ngram.evaluate(trigram, test)),
    }

    out["by_position"] = {"ngram_kn3": ngram_by_position(trigram, test, context_length)}
    device = pick_device(config["train"]["device"])
    inputs, targets = (t.to(device) for t in chunk(streams["test"], context_length, vocab.pad_id))
    for name in NEURAL:
        model, _ = load_checkpoint(REPO_ROOT / "runs" / name / "best.pt", device)
        out["by_position"][name] = neural_by_position(model, inputs, targets, vocab.pad_id)
        del model

    generations = json.load(open(REPO_ROOT / "results" / "generations.json"))["generations"]
    out["generations"] = {}
    for name, samples in generations.items():
        per_sample = [repetition(sample["continuation"].split()) for sample in samples]
        out["generations"][name] = {key: sum(s[key] for s in per_sample) / len(per_sample) for key in per_sample[0]}

    with open(REPO_ROOT / "results" / "analysis_part1.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/analysis_part1.json")


if __name__ == "__main__":
    main()
