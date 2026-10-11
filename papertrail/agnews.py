"""AG News for the Part III probe: subsets, tokenization, ids in the language-model vocabulary.

The language models only know WikiText-2's tokens, so each news item is
tokenized the way WikiText-2 is and then looked up in that vocabulary. Anything
it does not contain becomes <unk>, exactly as it would for the language model.

Print the subset sizes and the <unk> and truncation rates with:

    python -m papertrail.agnews configs/part3/probe.yaml
"""
import html
import re
import sys
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset

from papertrail.config import REPO_ROOT, load_config
from papertrail.data import Vocab, load_corpus

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# One token is, in order of preference: a WikiText marker, a dotted abbreviation
# (u.s., p.m.), an apostrophe suffix glued to the word before it ('s, 't,
# 'brien), a run of letters and digits, or any other single character.
TOKEN = re.compile(r"@[-,.]@|(?:[a-z]\.){2,}|(?<=\w)'\w+|\w+|[^\w\s]")


def tokenize(text: str, vocab: Vocab) -> list[str]:
    """Tokenize a news item the way WikiText-2 is tokenized.

    WikiText-2 is lowercased here, splits punctuation off words, writes
    "1,000" as "1 @,@ 000", "3.5" as "3 @.@ 5" and "well-known" as
    "well @-@ known", splits "don't" as "don 't", and keeps some dotted
    abbreviations such as "u.s." whole. This follows those conventions; it is
    an approximation of the original tool, not a copy.
    """
    # AG News carries scraping debris: HTML tags, entities that lost their "&"
    # ("#39;" for an apostrophe, "AT amp;T", "quot;", "nbsp;"), two single
    # quotes for an opening quote mark, and backslashes where lines were joined.
    text = re.sub(r"&lt;.*?&gt;", " ", text)
    text = re.sub(r"\s?&?#(\d+);", lambda match: chr(int(match.group(1))), text)
    text = re.sub(r"\s?&?amp;", "&", text)
    text = re.sub(r"&?quot;", '"', text)
    text = re.sub(r"&?nbsp;", " ", text)
    text = html.unescape(text).replace("''", '"').replace("\\", " ").lower()

    text = re.sub(r"(?<=\d),(?=\d)", " @,@ ", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", " @.@ ", text)
    text = re.sub(r"(?<=\w)-(?=\w)", " @-@ ", text)

    tokens = []
    for token in TOKEN.findall(text):
        # An abbreviation WikiText-2 does not have as one token ("u.k.") is
        # more use as its letters and dots than as a single <unk>.
        if len(token) > 3 and "." in token and token not in vocab.stoi:
            tokens.extend(re.findall(r"\w|\.", token))
        else:
            tokens.append(token)
    return tokens


@dataclass
class Split:
    """One subset of AG News, ready for a language model.

    ids:     (n, max_len) token ids, padded at the END with <pad>.
    lengths: (n,) the number of real tokens in each row, at least 1.
    labels:  (n,) class index, see CLASS_NAMES.
    texts:   the original, untokenized items, for reading the errors.
    """

    ids: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor
    texts: list[str]
    n_tokens: int           # before truncation
    n_unk: int              # before truncation
    n_truncated: int        # items longer than max_len


def encode(texts: list[str], labels: list[int], vocab: Vocab, max_len: int) -> Split:
    ids = torch.full((len(texts), max_len), vocab.pad_id, dtype=torch.long)
    lengths = torch.zeros(len(texts), dtype=torch.long)
    n_tokens = n_unk = n_truncated = 0

    for row, text in enumerate(texts):
        token_ids = vocab.encode(tokenize(text, vocab))
        n_tokens += len(token_ids)
        n_unk += token_ids.count(vocab.unk_id)
        n_truncated += len(token_ids) > max_len

        token_ids = token_ids[:max_len]
        ids[row, : len(token_ids)] = torch.tensor(token_ids)
        lengths[row] = len(token_ids)

    return Split(ids, lengths, torch.tensor(labels), texts, n_tokens, n_unk, n_truncated)


def load_agnews(config: dict, vocab: Vocab) -> dict[str, Split]:
    """The three subsets used by the probe: "train", "validation" and "test".

    Train and validation are disjoint, class-balanced samples of the official
    train split, drawn with the global seed. Test is the official test split,
    whole and untouched.
    """
    task = config["task"]
    dataset = load_dataset(task["dataset"], cache_dir=str(REPO_ROOT / config["data"]["cache_dir"]))
    assert dataset["train"].features["label"].names == CLASS_NAMES

    train_labels = np.array(dataset["train"]["label"])
    per_class = {"train": task["n_train"] // len(CLASS_NAMES), "validation": task["n_val"] // len(CLASS_NAMES)}
    rng = np.random.default_rng(config["seed"])
    chosen = {"train": [], "validation": []}
    for label in range(len(CLASS_NAMES)):
        # Shuffle this class once, then deal the first items to train and the
        # next ones to validation, so the two can never overlap.
        indices = rng.permutation(np.flatnonzero(train_labels == label))
        chosen["train"].extend(indices[: per_class["train"]])
        chosen["validation"].extend(indices[per_class["train"] : per_class["train"] + per_class["validation"]])

    splits = {}
    for name, indices in chosen.items():
        rows = dataset["train"].select(sorted(int(i) for i in indices))
        splits[name] = encode(rows["text"], rows["label"], vocab, task["max_len"])
    assert task["test"] == "full"
    splits["test"] = encode(dataset["test"]["text"], dataset["test"]["label"], vocab, task["max_len"])
    return splits


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    vocab, _ = load_corpus(config)
    splits = load_agnews(config, vocab)

    for name, split in splits.items():
        per_class = torch.bincount(split.labels, minlength=len(CLASS_NAMES)).tolist()
        print(
            f"{name:>10}: {len(split.texts):>5} items  per class {per_class}  "
            f"mean length {split.n_tokens / len(split.texts):.1f}  "
            f"<unk> {split.n_unk / split.n_tokens:.2%}  "
            f"longer than {config['task']['max_len']}: {split.n_truncated / len(split.texts):.2%}"
        )
    example = splits["train"]
    print("text:  ", example.texts[0])
    print("tokens:", " ".join(vocab.decode(example.ids[0, : example.lengths[0]].tolist())))
