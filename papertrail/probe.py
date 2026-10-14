"""Part III: do the language models' hidden states carry topic information?

    python -m papertrail.probe --config configs/part3/probe.yaml

For every trained language model and every pooling method, each AG News item
is turned into one vector and a logistic-regression classifier is trained on
those vectors. The language models are frozen throughout; only the classifier
learns. Writes results/<run_name>.json.
"""
import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from papertrail.agnews import CLASS_NAMES, Split, load_agnews
from papertrail.config import REPO_ROOT, load_config
from papertrail.data import load_corpus
from papertrail.models import load_checkpoint
from papertrail.pooling import last_pool, mean_pool
from papertrail.utils import pick_device, set_seed

POOLING = {"mean_nonpad": mean_pool, "last_nonpad": last_pool}
BATCH_SIZE = 256


def extract(model: torch.nn.Module, split: Split, device: torch.device) -> dict[str, np.ndarray]:
    """One vector per item, for every pooling method: {pooling: (n, width) array}.

    The padding sits at the end of each row and both models read left to
    right, so the hidden states at the real positions are exactly what they
    would be without padding. The pooling functions then ignore the rest.
    """
    features = {name: [] for name in POOLING}
    with torch.no_grad():
        for start in range(0, len(split.ids), BATCH_SIZE):
            ids = split.ids[start:start + BATCH_SIZE].to(device)
            lengths = split.lengths[start:start + BATCH_SIZE].to(device)
            hidden, _ = model.features(ids)          # (batch, time, width), final layer
            for name, pool in POOLING.items():
                features[name].append(pool(hidden, lengths).cpu())
    return {name: torch.cat(batches).numpy() for name, batches in features.items()}


def fit_probe(features: dict[str, np.ndarray], labels: dict[str, np.ndarray], classifier: dict) -> dict:
    """Train the classifier, choose its regularization on validation, test once.

    features, labels: one array per subset ("train", "validation", "test").

    The features are standardized first (zero mean, unit variance per
    dimension, using training statistics only). The two models' hidden states
    live on different scales, and without this the same C would regularize
    them by different amounts.
    """
    candidates = []
    for C in classifier["C_grid"]:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=classifier["max_iter"]))
        model.fit(features["train"], labels["train"])
        predicted = model.predict(features["validation"])
        candidates.append({
            "C": C,
            "val_accuracy": accuracy_score(labels["validation"], predicted),
            "val_macro_f1": f1_score(labels["validation"], predicted, average="macro"),
            "model": model,
        })

    # The test set plays no part in this choice; it is touched once, below.
    best = max(candidates, key=lambda candidate: candidate[classifier["select_on"]])
    probabilities = best["model"].predict_proba(features["test"])
    predicted = probabilities.argmax(axis=1)

    return {
        "C": best["C"],
        "val_accuracy": best["val_accuracy"],
        "val_macro_f1": best["val_macro_f1"],
        "test_accuracy": accuracy_score(labels["test"], predicted),
        "test_macro_f1": f1_score(labels["test"], predicted, average="macro"),
        "per_class_f1": dict(zip(CLASS_NAMES, f1_score(labels["test"], predicted, average=None).tolist())),
        # Rows are the true class, columns the predicted class, in CLASS_NAMES order.
        "confusion_matrix": confusion_matrix(labels["test"], predicted).tolist(),
        "validation_by_C": [{k: v for k, v in c.items() if k != "model"} for c in candidates],
        "predicted": predicted,
        "confidence": probabilities.max(axis=1),
    }


def error_examples(result: dict, split: Split, n: int) -> list[dict]:
    """The n test items the classifier got wrong with the highest confidence.

    Confident mistakes are the informative ones: the representation did not
    just leave the item ambiguous, it pointed firmly at the wrong topic.
    """
    labels = split.labels.numpy()
    wrong = np.flatnonzero(result["predicted"] != labels)
    most_confident = wrong[np.argsort(-result["confidence"][wrong])][:n]
    return [
        {
            "true": CLASS_NAMES[labels[i]],
            "predicted": CLASS_NAMES[result["predicted"][i]],
            "confidence": float(result["confidence"][i]),
            "tokens": int(split.lengths[i]),
            "text": split.texts[i],
        }
        for i in most_confident
    ]


def unk_shares(split: Split, unk_id: int) -> np.ndarray:
    """The share of each item's real tokens that are <unk>."""
    real = torch.arange(split.ids.size(1))[None, :] < split.lengths[:, None]
    return (((split.ids == unk_id) & real).sum(1).float() / split.lengths.float()).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = pick_device(config["train"]["device"])
    vocab, _ = load_corpus(config)
    splits = load_agnews(config, vocab)
    labels = {name: split.labels.numpy() for name, split in splits.items()}
    test_unk = unk_shares(splits["test"], vocab.unk_id)

    results = {
        "run_name": config["run_name"],
        "task": config["task"],
        "classifier": config["classifier"],
        "classes": CLASS_NAMES,
        "data": {
            name: {
                "items": len(split.texts),
                "unk_rate": split.n_unk / split.n_tokens,
                "truncated": split.n_truncated / len(split.texts),
            }
            for name, split in splits.items()
        },
        "unk_share_by_class": {
            name: float(test_unk[labels["test"] == c].mean()) for c, name in enumerate(CLASS_NAMES)
        },
        "probes": [],
    }

    for checkpoint_path in config["representations"]["models"]:
        model, checkpoint = load_checkpoint(REPO_ROOT / checkpoint_path, device)
        lm_name = checkpoint["config"]["run_name"]
        # The language model's own test perplexity, to set beside the probe
        # scores: does the better language model give the better features?
        lm_results = json.load(open(REPO_ROOT / config["output"]["results_dir"] / f"{lm_name}.json"))

        features = {name: extract(model, split, device) for name, split in splits.items()}
        for pooling in config["representations"]["pooling"]:
            result = fit_probe({name: features[name][pooling] for name in splits}, labels, config["classifier"])
            errors = error_examples(result, splits["test"], config["n_error_examples"])
            predicted = result.pop("predicted")
            n_wrong = int((predicted != labels["test"]).sum())
            result.pop("confidence")
            # Are the items confused between Business and Sci/Tech richer in
            # <unk> than the ones of the same class classified correctly?
            business, sci_tech = CLASS_NAMES.index("Business"), CLASS_NAMES.index("Sci/Tech")
            result["unk_share_business_scitech"] = {
                f"{CLASS_NAMES[true]}_{'correct' if pred == true else 'as_' + CLASS_NAMES[pred]}": float(
                    test_unk[(labels["test"] == true) & (predicted == pred)].mean()
                )
                for true, other in ((business, sci_tech), (sci_tech, business))
                for pred in (true, other)
            }
            results["probes"].append({
                "model": lm_name,
                "pooling": pooling,
                "width": int(features["train"][pooling].shape[1]),
                "lm_test_perplexity": lm_results["perplexity"]["test"],
                **result,
                "test_errors": n_wrong,
                "error_examples": errors,
            })
            print(
                f"{lm_name:>12} {pooling:<12} C={result['C']:<5} "
                f"val F1 {result['val_macro_f1']:.4f}  "
                f"test accuracy {result['test_accuracy']:.4f}  test macro-F1 {result['test_macro_f1']:.4f}"
            )

    path = REPO_ROOT / config["output"]["results_dir"] / f"{config['run_name']}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
