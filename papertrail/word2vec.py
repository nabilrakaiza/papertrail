"""Train skip-gram word vectors on the WikiText-2 train split (Part II, setting 2).

    python -m papertrail.word2vec --config configs/part2/word2vec.yaml

Writes the vectors to the path in `word2vec.output` (under embeddings/, which is
gitignored) and the training cost to results/<run_name>.json. The cost is
reported on its own, separately from the language-model training that later
uses these vectors.
"""
import argparse
import json
import time

from datasets import load_dataset
from gensim.models import Word2Vec

from papertrail.config import REPO_ROOT, load_config
from papertrail.data import tokenize


def load_sentences(config: dict, split: str) -> list[list[str]]:
    """One list of tokens per non-empty line of a split.

    Each line goes through the same `tokenize` as the language-model data, so
    the tokens, the lowercasing and the closing <eos> are identical. The lines
    are kept apart, unlike the single stream the language models read, so that
    a skip-gram context window never reaches across a line break.
    """
    data = config["data"]
    dataset = load_dataset(data["dataset"], data["config"], cache_dir=str(REPO_ROOT / data["cache_dir"]))
    sentences = (
        tokenize([line], data["lowercase"], data["add_eos"], data["drop_empty_lines"])
        for line in dataset[split]["text"]
    )
    return [sentence for sentence in sentences if sentence]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["word2vec"]
    assert settings["method"] == "skipgram", "only skip-gram is implemented"
    # Validation and test text must not shape the vectors: that would leak
    # held-out data into the language model through its input embeddings.
    assert settings["train_split_only"], "the vectors must come from the train split alone"

    sentences = load_sentences(config, "train")
    n_tokens = sum(len(sentence) for sentence in sentences)

    start = time.perf_counter()
    model = Word2Vec(
        sentences=sentences,
        sg=1,                                  # 1 = skip-gram, 0 = CBOW
        vector_size=settings["vector_size"],
        window=settings["window"],
        negative=settings["negative"],         # negative samples per positive pair
        min_count=settings["min_count"],
        epochs=settings["epochs"],
        workers=settings["workers"],           # 1: with more threads the result depends on timing
        seed=config["seed"],
    )
    train_seconds = time.perf_counter() - start

    output = REPO_ROOT / settings["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(output))

    results = {
        "run_name": config["run_name"],
        "settings": settings,
        # Everything else is left at gensim's defaults; record the ones that matter.
        "gensim_defaults": {"sample": model.sample, "alpha": model.alpha, "ns_exponent": model.ns_exponent},
        "sentences": len(sentences),
        "tokens": n_tokens,
        "vocabulary": len(model.wv),
        "train_seconds": train_seconds,
        "device": "cpu",
    }
    results_path = REPO_ROOT / config["output"]["results_dir"] / f"{config['run_name']}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{len(model.wv):,} vectors of size {model.vector_size} from {n_tokens:,} tokens in {train_seconds:.1f}s")
    print(f"wrote {output} and {results_path}")


if __name__ == "__main__":
    main()
