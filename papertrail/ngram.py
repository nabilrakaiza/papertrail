"""Interpolated Kneser-Ney trigram language model.

    python -m papertrail.ngram --config configs/part1/ngram.yaml
"""
import argparse
import json
import math
import time
from dataclasses import dataclass

import torch

from papertrail.config import REPO_ROOT, load_config
from papertrail.data import load_corpus


# ---------------------------------------------------------------------------
# 1. Counting
# ---------------------------------------------------------------------------

@dataclass
class NgramCounts:
    """Everything the Kneser-Ney formula reads from the training data.

    u, v, w are three consecutive ids. A dot stands for "any token", and
    N(...) counts distinct tokens that can fill the dot, not occurrences.
    """

    # Keyed by a triple (u, v, w)
    trigram: dict[tuple[int, int, int], int]             # c(u,v,w): times the triple occurs

    # Keyed by a pair
    context: dict[tuple[int, int], int]                  # c(u,v): times (u, v) is followed by a token
    context_followers: dict[tuple[int, int], int]        # distinct w seen after (u, v)
    bigram_preceders: dict[tuple[int, int], int]         # N(.,v,w): distinct u seen before (v, w)

    # Keyed by a single id
    middle_total: dict[int, int]                         # N(.,v,.): sum of N(.,v,w) over w
    token_followers: dict[int, int]                      # distinct w with N(.,v,w) > 0
    token_preceders: dict[int, int]                      # N(.,w): distinct v seen before w

    # A single number
    n_bigram_types: int                                  # N(.,.): distinct pairs in the corpus


def count_ngrams(stream: list[int]) -> NgramCounts:
    """Count everything the model needs from the train ids.

    Pass a list of Python ints (`streams["train"].tolist()`), not the tensor.
    Elements of a tensor are themselves tensors and do not work as dict keys.
    """
    N = len(stream)

    # c(u,v,w)
    trigram = {}
    for i in range(N-2):
        tup = (stream[i], stream[i+1], stream[i+2])
        if tup not in trigram:
            trigram[tup] = 0
        trigram[tup] += 1

    # c(u,v). Stops at N-2, not N-1: the last pair of the stream has no token
    # after it, and counting it would make this larger than the sum of c(u,v,w).
    context = {}
    for i in range(N-2):
        tup = (stream[i], stream[i+1])
        if tup not in context:
            context[tup] = 0
        context[tup] += 1

    # Distinct w seen after (u, v)
    temp = {}
    for i in range(N-2):
        tup = (stream[i], stream[i+1])
        if tup not in temp:
            temp[tup] = set()
        temp[tup].add(stream[i+2])

    context_followers = {k: len(temp[k]) for k in temp.keys()}

    # N(.,v,w): distinct u seen before (v, w). Each distinct triple (u, v, w)
    # contributes exactly one u to the pair (v, w), so no set is needed.
    bigram_preceders = {}
    for (u, v, w) in trigram:
        if (v, w) not in bigram_preceders:
            bigram_preceders[(v, w)] = 0
        bigram_preceders[(v, w)] += 1

    # N(.,v,.) and the number of distinct w with N(.,v,w) > 0, both per v
    middle_total = {}
    token_followers = {}
    for (v, w) in bigram_preceders:
        if v not in middle_total:
            middle_total[v] = 0
            token_followers[v] = 0
        middle_total[v] += bigram_preceders[(v, w)]
        token_followers[v] += 1

    # Distinct pairs anywhere in the stream. This one does run to N-1: the last
    # pair has no follower, but its second token still has a preceder.
    bigram_types = set()
    for i in range(N-1):
        bigram_types.add((stream[i], stream[i+1]))

    # N(.,w): distinct v seen before w
    token_preceders = {}
    for (v, w) in bigram_types:
        if w not in token_preceders:
            token_preceders[w] = 0
        token_preceders[w] += 1

    return NgramCounts(
        trigram=trigram,
        context=context,
        context_followers=context_followers,
        bigram_preceders=bigram_preceders,
        middle_total=middle_total,
        token_followers=token_followers,
        token_preceders=token_preceders,
        n_bigram_types=len(bigram_types),
    )


# ---------------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------------

class KneserNeyTrigram:
    """Holds the count tables and the discount, and answers probability queries."""

    def __init__(self, counts: NgramCounts, discount: float, vocab_size: int):
        self.counts = counts 
        self.discount = discount 
        self.vocab_size = vocab_size

    def prob(self, token: int, context: tuple[int, ...]) -> float:
        """P(token | context), where context is the previous one or two ids.

        Used for perplexity. Cases to handle:
          - a full two-token context seen in training;
          - a context never seen in training (nothing to discount);
          - a one-token context, at the start of a stream.
        """
        if len(context) == 2:
            if context in self.counts.context:
                num = max(self.counts.trigram.get((context[0], context[1], token),0) - self.discount, 0)
                denom = self.counts.context.get(context)
                lamb = self.discount / (self.counts.context[context]) * self.counts.context_followers[context]
                pcont = self.bigram_prob(token, context[1])

                prob = num/denom + lamb * pcont

            else:
                # Unseen context: nothing was counted, so nothing was discounted,
                # and the level below gets all of the probability.
                prob = self.bigram_prob(token, context[1])

        else:
            # One token of context (the first target of a stream): start one level down.
            prob = self.bigram_prob(token, context[0])

        return prob

    def bigram_prob(self, token: int, v: int) -> float:
        """Pcont(token | v): the bigram level.

        Same shape as the trigram level, but built on counts of distinct
        contexts rather than counts of occurrences.
        """
        pcont = self.unigram_prob(token)

        if v in self.counts.middle_total:
            num = max(self.counts.bigram_preceders.get((v, token), 0) - self.discount, 0)
            denom = self.counts.middle_total[v]
            lamb = self.discount / denom * self.counts.token_followers[v]

            prob = num/denom + lamb * pcont

        else:
            prob = pcont

        return prob

    def unigram_prob(self, token: int) -> float:
        """Pcont(token): the share of all distinct pairs that end in this token."""
        return self.counts.token_preceders.get(token, 0) / self.counts.n_bigram_types

    def distribution(self, context: tuple[int, ...]) -> torch.Tensor:
        """P(. | context) for every token, as a tensor of shape (vocab_size,).

        Used for generation and for the timing benchmark. Must sum to 1.
        """
        tensor = torch.zeros(self.vocab_size)

        for i in range(self.vocab_size):
            tensor[i] = self.prob(i, context)

        return tensor


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: KneserNeyTrigram, stream: list[int]) -> float:
    """Mean negative log-probability (in nats) over the targets of a stream.

    The targets are every token except the first, the same ones the neural
    models are scored on. Perplexity is exp of the returned value.
    """

    total = -1 * math.log(model.prob(stream[1], (stream[0],)))

    for i in range(2, len(stream)):
        prob = model.prob(stream[i], (stream[i-2], stream[i-1]))
        total -= math.log(prob)

    total /= len(stream) - 1

    return total


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    vocab, streams = load_corpus(config)
    streams = {split: stream.tolist() for split, stream in streams.items()}

    # Fit. For an n-gram model, "training" is counting.
    start = time.perf_counter()
    counts = count_ngrams(streams["train"])
    model = KneserNeyTrigram(counts, config["model"]["discount"], len(vocab))
    fit_seconds = time.perf_counter() - start

    # Sanity check: a seen context, an unseen one, and a one-token one must each
    # give a distribution that sums to 1 (up to float32 rounding).
    train = streams["train"]
    for context in [(train[0], train[1]), (vocab.pad_id, vocab.pad_id), (train[0],)]:
        total = model.distribution(context).sum().item()
        assert abs(total - 1) < 1e-4, f"distribution for {context} sums to {total}"

    results = {
        "run_name": config["run_name"],
        "model": config["model"],
        "vocab_size": len(vocab),
        "fit_seconds": fit_seconds,
        # The n-gram's "model size": what it has to store.
        "stored_ngrams": {
            "trigrams": len(counts.trigram),
            "bigrams": counts.n_bigram_types,
            "unigrams": len(counts.token_preceders),
        },
        "loss": {},
        "perplexity": {},
    }
    for split in ("validation", "test"):
        loss = evaluate(model, streams[split])
        results["loss"][split] = loss
        results["perplexity"][split] = math.exp(loss)
        print(f"{split:>10}: loss {loss:.4f}  perplexity {math.exp(loss):.2f}")
    print(f"fit in {fit_seconds:.1f}s, {len(counts.trigram):,} distinct trigrams")

    results_dir = REPO_ROOT / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{config['run_name']}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
