"""Generate text from every model with the same prompts and decoding settings.

    python -m papertrail.generate --configs configs/part1/*.yaml

The prompts and settings (temperature, top-k, length, seed) come from the
`generate` section of the config. Writes results/generations.json, or the file
given with --output.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from papertrail import ngram
from papertrail.config import REPO_ROOT, load_config
from papertrail.data import load_corpus
from papertrail.models import load_checkpoint
from papertrail.utils import pick_device


# ---------------------------------------------------------------------------
# 1. Sampling: one function, shared by all four models
# ---------------------------------------------------------------------------

def sample_next(log_probs: torch.Tensor, temperature: float, top_k: int, generator: torch.Generator) -> int:
    """Choose the next token id from a model's scores.

    log_probs: (vocab_size,) log-probabilities for the next token, on the CPU.
    generator: a seeded torch.Generator, so the draw is reproducible.

    Steps, in this order:
      1. Temperature: divide the log-probabilities by `temperature`.
      2. Top-k: keep the `top_k` highest values and rule out all the others.
      3. Turn what is left back into probabilities that sum to 1.
      4. Draw one token id at random from them, using `generator`.
    """

    divided_log_probs = log_probs / temperature
    vals, idx = torch.topk(divided_log_probs, k=top_k)
    probs = torch.softmax(vals, dim=-1)
    choice = torch.multinomial(probs, num_samples=1, generator=generator)

    return idx.gather(-1, choice).item()


# ---------------------------------------------------------------------------
# 2. Next-token scores from each kind of model
# ---------------------------------------------------------------------------

def neural_log_probs(model: torch.nn.Module, ids: list[int], device: torch.device) -> torch.Tensor:
    """Log-probabilities for the token that follows `ids`, from an RNN, LSTM or Transformer.

    Run the whole sequence so far through the model, without a state, and keep
    the scores at the last position. The model expects a batch, shape
    (1, length), on its own device, and returns (logits, state) with logits of
    shape (1, length, vocab_size).

    Returns a (vocab_size,) tensor on the CPU.
    """
    inputs = torch.tensor([ids], device=device)
    logits, state = model(inputs)
    temp = logits[0, -1]
    
    res = F.log_softmax(temp, dim = -1)
    res = res.cpu()

    return res


def ngram_log_probs(model, ids: list[int]) -> torch.Tensor:
    """Log-probabilities for the token that follows `ids`, from the trigram model.

    `model.distribution(context)` gives probabilities, given the last two ids
    as a tuple. A probability of exactly 0 (only <pad>) becomes -inf under log,
    which is fine: it means "never choose this".

    Returns a (vocab_size,) tensor on the CPU.
    """
    # The trigram wants its context as a tuple of plain ids: the last two, or
    # the only one when the prompt is a single token.
    context = tuple(ids[-2:])
    prob = model.distribution(context)
    # These are already probabilities, so log is enough; log_softmax is for logits.
    res = torch.log(prob)

    return res


# ---------------------------------------------------------------------------
# 3. The generation loop
# ---------------------------------------------------------------------------

def generate(next_log_probs, prompt_ids: list[int], settings: dict) -> list[int]:
    """Continue a prompt by `settings["max_new_tokens"]` tokens.

    next_log_probs: a function that takes the ids so far and returns the
                    (vocab_size,) log-probabilities for the next token. Passing
                    it in is what lets one loop serve all four models.
    settings:       the `generate` section of the config.

    Create the generator here, seeded with settings["seed"], so that every
    continuation starts from the same random state whichever model is used.
    Keep going past <eos>: every continuation must have the same length.

    Returns only the new ids, not the prompt.
    """
    g = torch.Generator()
    g.manual_seed(settings["seed"])

    # A copy, so the caller's prompt is not changed. Each new token is appended
    # and becomes part of the context for the next one.
    ids = list(prompt_ids)
    for _ in range(settings["max_new_tokens"]):
        log_probs = next_log_probs(ids)
        ids.append(sample_next(log_probs, settings["temperature"], settings["top_k"], g))

    return ids[len(prompt_ids):]


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="results/generations.json")
    args = parser.parse_args()

    configs = [load_config(path) for path in args.configs]
    # Every run shares one corpus and one set of prompts and decoding settings
    # (configs/base.yaml), which is what makes the samples comparable.
    settings = configs[0]["generate"]
    vocab, streams = load_corpus(configs[0])

    results = {}
    for config in configs:
        if config["model"]["type"] == "ngram":
            counts = ngram.count_ngrams(streams["train"].tolist())
            model = ngram.KneserNeyTrigram(counts, config["model"]["discount"], len(vocab))
            next_log_probs = lambda ids: ngram_log_probs(model, ids)
        else:
            device = pick_device(config["train"]["device"])
            path = REPO_ROOT / config["output"]["runs_dir"] / config["run_name"] / "best.pt"
            model, _ = load_checkpoint(path, device)      # comes back in evaluation mode
            next_log_probs = lambda ids: neural_log_probs(model, ids, device)

        print(f"\n== {config['run_name']}")
        results[config["run_name"]] = []
        for prompt in settings["prompts"]:
            prompt_ids = vocab.encode(prompt.split())
            assert vocab.unk_id not in prompt_ids, f"prompt has a word outside the vocabulary: {prompt!r}"
            with torch.no_grad():
                new_ids = generate(next_log_probs, prompt_ids, settings)
            continuation = " ".join(vocab.decode(new_ids))
            results[config["run_name"]].append({"prompt": prompt, "continuation": continuation})
            print(f"[{prompt}] {continuation}")

    output = {
        "settings": {key: value for key, value in settings.items() if key != "prompts"},
        "generations": results,
    }
    with open(REPO_ROOT / args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
