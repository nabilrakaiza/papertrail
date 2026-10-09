"""WikiText-2 data pipeline: tokens, vocabulary, and chunks for the language models.

The steps follow the "Data" section of the README:

    raw lines -> lowercase, split on whitespace, <eos> per line -> one token stream per split
    train stream -> vocabulary
    token stream -> stream of ids -> (inputs, targets) chunks of `context_length`

Print corpus statistics with:

    python -m papertrail.data configs/base.yaml
"""
import sys
from collections import Counter
from collections.abc import Iterator

import torch
from datasets import load_dataset

from papertrail.config import REPO_ROOT, load_config

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
SPLITS = ("train", "validation", "test")


class Vocab:
    """A fixed mapping between tokens and integer ids."""

    def __init__(self, tokens: list[str], specials: list[str]):
        # Specials come first so their ids do not depend on the corpus (<pad> is 0).
        # The rest are ordered by frequency, ties broken alphabetically. The order
        # has to be deterministic: a seeded embedding table assigns row i to token i,
        # so a different order would mean a different initialisation.
        counts = Counter(tokens)
        for special in specials:
            counts.pop(special, None)
        by_frequency = sorted(counts, key=lambda token: (-counts[token], token))

        self.itos = list(specials) + by_frequency
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.pad_id, self.unk_id, self.eos_id = self.stoi[PAD], self.stoi[UNK], self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        """Map tokens to ids. Anything not in the vocabulary becomes <unk>."""
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]


def tokenize(lines: list[str], lowercase: bool, add_eos: bool, drop_empty_lines: bool) -> list[str]:
    """Turn the lines of one split into a single flat list of tokens.

    WikiText-2-v1 is already tokenized, so splitting on whitespace is enough.
    Section headings such as "= = gameplay = =" are kept as ordinary text.
    """
    tokens = []
    for line in lines:
        if lowercase:
            line = line.lower()
        words = line.split()
        if drop_empty_lines and not words:
            continue
        tokens.extend(words)
        if add_eos:
            tokens.append(EOS)
    return tokens


def load_corpus(config: dict) -> tuple[Vocab, dict[str, torch.Tensor]]:
    """Download (or read from cache) the corpus and encode every split.

    Returns the vocabulary and, for each of "train", "validation" and "test",
    a 1-D LongTensor holding that split as one stream of token ids.
    """
    data = config["data"]
    dataset = load_dataset(data["dataset"], data["config"], cache_dir=str(REPO_ROOT / data["cache_dir"]))

    tokens = {
        split: tokenize(dataset[split]["text"], data["lowercase"], data["add_eos"], data["drop_empty_lines"])
        for split in SPLITS
    }
    # Train split only: a validation or test word that never occurs in training
    # must look unknown to the model, otherwise the test set leaks into it.
    vocab = Vocab(tokens[data["vocab_from"]], data["special_tokens"])
    streams = {split: torch.tensor(vocab.encode(tokens[split]), dtype=torch.long) for split in SPLITS}
    return vocab, streams


def chunk(stream: torch.Tensor, context_length: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut a stream of ids into contiguous (inputs, targets) chunks.

    Both tensors have shape (n_chunks, context_length). The target at each
    position is the token that follows the input at that position:

        stream   t0 t1 t2 t3 t4 t5 t6          context_length = 3
        inputs   [t0 t1 t2] [t3 t4 t5]
        targets  [t1 t2 t3] [t4 t5 t6]

    Every token except the first is a target exactly once. If the stream does
    not divide evenly, the final chunk is filled with <pad>, which the loss
    must ignore (`ignore_index=pad_id`).
    """
    inputs, targets = stream[:-1], stream[1:]
    n_pad = -len(inputs) % context_length
    padding = torch.full((n_pad,), pad_id, dtype=stream.dtype)
    inputs = torch.cat([inputs, padding]).view(-1, context_length)
    targets = torch.cat([targets, padding]).view(-1, context_length)
    return inputs, targets


def sequential_batches(
    inputs: torch.Tensor, targets: torch.Tensor, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches in which each row continues the same row of the previous batch.

    The chunks are dealt into `batch_size` parallel tracks, each a contiguous
    stretch of the corpus, and one step is read from every track at a time.
    With 6 chunks c0..c5 and batch_size 2:

        track 0:  c0 c1 c2          batch 0 = [c0, c3]
        track 1:  c3 c4 c5          batch 1 = [c1, c4]
                                    batch 2 = [c2, c5]

    This is what lets an RNN or LSTM carry its hidden state from one batch to
    the next (truncated backpropagation through time): row b's state after
    batch k is the right starting state for row b of batch k + 1.

    Chunks that do not fill a whole track are dropped (fewer than `batch_size`
    of them), so use this for training only. For evaluation, iterate over all
    chunks so that every target is counted.
    """
    n_batches = len(inputs) // batch_size
    n_used = n_batches * batch_size
    inputs = inputs[:n_used].view(batch_size, n_batches, -1)
    targets = targets[:n_used].view(batch_size, n_batches, -1)
    for step in range(n_batches):
        yield inputs[:, step], targets[:, step]


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    vocab, streams = load_corpus(config)
    context_length = config["train"]["context_length"]

    print(f"vocabulary: {len(vocab):,} tokens (including {', '.join(config['data']['special_tokens'])})")
    for split, stream in streams.items():
        inputs, _ = chunk(stream, context_length, vocab.pad_id)
        unk_rate = (stream == vocab.unk_id).float().mean().item()
        print(f"{split:>10}: {len(stream):>9,} tokens  {len(inputs):>6,} chunks of {context_length}  <unk> {unk_rate:.2%}")
    print("first tokens of train:", " ".join(vocab.decode(streams["train"][:20].tolist())))
