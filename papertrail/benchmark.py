"""Inference throughput and model size, on one common workload.

    python -m papertrail.benchmark --configs configs/part1/*.yaml

The workload is the test split, cut into chunks of `context_length` and read in
batches of `batch_size` (both from the `benchmark` section of the config).
Throughput is test targets per second. Writes results/benchmark.json.

Every neural model computes a full distribution over the vocabulary at every
position, whether or not all of it is needed. The n-gram model does not have
to: scoring a known next token is a few dictionary lookups. So the n-gram gets
two numbers, one for each kind of work:

    score_targets       the probability of each actual next token
                        (all that perplexity needs)
    full_distribution   the probability of every vocabulary token at each
                        position (what generation needs, and what the neural
                        models are timed on)
"""
import argparse
import json
import time

import torch

from papertrail import ngram
from papertrail.config import REPO_ROOT, load_config
from papertrail.data import chunk, load_corpus
from papertrail.models import count_parameters, load_checkpoint
from papertrail.utils import pick_device, synchronize

# The n-gram's full distribution takes about 0.1 s per position in pure
# Python, so it is timed on the first positions of the test split only. The
# whole split would take hours.
NGRAM_DISTRIBUTION_POSITIONS = 256


def time_neural(model: torch.nn.Module, inputs: torch.Tensor, n_targets: int, bench: dict, device: torch.device) -> list[float]:
    """Targets per second for full forward passes over `inputs`, one value per repeat."""
    batch_size = bench["batch_size"]
    batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
    rates = []
    with torch.no_grad():
        # The first batches on a GPU pay one-off costs (memory allocation,
        # kernel compilation) that say nothing about steady-state speed.
        for batch in batches[: bench["warmup_batches"]]:
            model(batch)
            synchronize(device)

        for _ in range(bench["repeats"]):
            start = time.perf_counter()
            for batch in batches:
                model(batch)
                # Wait for this batch before submitting the next. A GPU call
                # only queues work; without the wait, 120 batches of logits
                # (0.24 GB each) pile up in memory faster than they are
                # computed. On the 8 GB machine that made the LSTM swap and
                # run at half speed from the second repeat on.
                synchronize(device)
            rates.append(n_targets / (time.perf_counter() - start))
    return rates


def benchmark_neural(config: dict, vocab, test: torch.Tensor) -> dict:
    bench = config["benchmark"]
    device = pick_device(config["train"]["device"])
    path = REPO_ROOT / config["output"]["runs_dir"] / config["run_name"] / "best.pt"
    model, _ = load_checkpoint(path, device)

    inputs, targets = chunk(test, bench["context_length"], vocab.pad_id)
    n_targets = (targets != vocab.pad_id).sum().item()
    rates = time_neural(model, inputs.to(device), n_targets, bench, device)
    return {
        "device": str(device),
        "parameters": count_parameters(model),
        "checkpoint_mb": path.stat().st_size / 1e6,
        "targets": n_targets,
        "tokens_per_second": {"full_distribution": sum(rates) / len(rates)},
        "repeats": {"full_distribution": rates},
    }


def benchmark_ngram(config: dict, vocab, train: list[int], test: list[int]) -> dict:
    bench = config["benchmark"]
    counts = ngram.count_ngrams(train)
    model = ngram.KneserNeyTrigram(counts, config["model"]["discount"], len(vocab))

    score_rates, distribution_rates = [], []
    for _ in range(bench["repeats"]):
        start = time.perf_counter()
        ngram.evaluate(model, test)
        score_rates.append((len(test) - 1) / (time.perf_counter() - start))

        start = time.perf_counter()
        for i in range(2, 2 + NGRAM_DISTRIBUTION_POSITIONS):
            model.distribution((test[i - 2], test[i - 1]))
        distribution_rates.append(NGRAM_DISTRIBUTION_POSITIONS / (time.perf_counter() - start))

    return {
        "device": "cpu",
        "stored_ngrams": {
            "trigrams": len(counts.trigram),
            "bigrams": counts.n_bigram_types,
            "unigrams": len(counts.token_preceders),
        },
        "targets": {"score_targets": len(test) - 1, "full_distribution": NGRAM_DISTRIBUTION_POSITIONS},
        "tokens_per_second": {
            "score_targets": sum(score_rates) / len(score_rates),
            "full_distribution": sum(distribution_rates) / len(distribution_rates),
        },
        "repeats": {"score_targets": score_rates, "full_distribution": distribution_rates},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    args = parser.parse_args()

    configs = [load_config(path) for path in args.configs]
    # Every run shares one corpus and one benchmark workload (configs/base.yaml).
    vocab, streams = load_corpus(configs[0])

    results = {}
    for config in configs:
        if config["model"]["type"] == "ngram":
            result = benchmark_ngram(config, vocab, streams["train"].tolist(), streams["test"].tolist())
        else:
            result = benchmark_neural(config, vocab, streams["test"])
        results[config["run_name"]] = {"workload": config["benchmark"], **result}
        rates = "  ".join(f"{kind} {rate:,.0f} tok/s" for kind, rate in result["tokens_per_second"].items())
        print(f"{config['run_name']:>12} on {result['device']}: {rates}")

    path = REPO_ROOT / configs[0]["output"]["results_dir"] / "benchmark.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
