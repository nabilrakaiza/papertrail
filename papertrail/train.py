"""Train a neural language model.

    python -m papertrail.train --config configs/part1/lstm.yaml

Writes:
    runs/<run_name>/best.pt     the checkpoint with the lowest validation loss
    runs/<run_name>/log.json    loss and time for every epoch
    results/<run_name>.json     final metrics, measured with best.pt
"""
import argparse
import json
import math
import time

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

from papertrail.config import REPO_ROOT, load_config
from papertrail.data import SPLITS, chunk, load_corpus, sequential_batches
from papertrail.embeddings import init_embeddings, input_embedding
from papertrail.evaluate import evaluate
from papertrail.models import build_model, count_parameters
from papertrail.rnn import detach
from papertrail.utils import pick_device, set_seed, synchronize


def make_schedule(optimizer: torch.optim.Optimizer, train_config: dict, total_steps: int) -> LambdaLR:
    """Learning-rate schedule: optional linear warmup, then optional cosine decay.

    A config with neither `warmup_steps` nor `scheduler` gets a constant rate.
    The schedule is stepped once per batch, not once per epoch.
    """
    warmup_steps = train_config.get("warmup_steps", 0)
    cosine = train_config.get("scheduler") == "cosine"

    def scale(step: int) -> float:
        # Warmup: rise linearly from near 0 to the full rate. Post-LN
        # Transformers diverge without it.
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        # Cosine: fall smoothly from the full rate to 0 at the last step.
        if cosine:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0

    return LambdaLR(optimizer, scale)


def train_epoch(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    schedule: LambdaLR,
    train_config: dict,
    pad_id: int,
) -> float:
    """One pass over the training chunks. Returns the mean loss per target.

    The returned loss is averaged over the epoch while the weights are still
    changing and dropout is on, so it is not directly comparable to a
    validation loss, which is measured afterwards with dropout off.
    """
    model.train()
    state = None
    loss_sum, n_targets = 0.0, 0

    for batch_inputs, batch_targets in sequential_batches(inputs, targets, train_config["batch_size"]):
        logits, state = model(batch_inputs, state)
        # cross_entropy wants (N, vocab) against (N,), so batch and time are
        # flattened together: every position is one classification.
        loss = F.cross_entropy(logits.flatten(0, 1), batch_targets.flatten(), ignore_index=pad_id)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
        optimizer.step()
        schedule.step()

        # Truncated backpropagation through time. The next batch continues
        # each row of this one (see sequential_batches), so a recurrent model
        # starts from the state it ended on, cut off from this batch's graph.
        # A Transformer returns no state, so there is nothing to carry.
        if state is not None and train_config["carry_hidden_train"]:
            state = detach(state)
        else:
            state = None

        batch_n = (batch_targets != pad_id).sum().item()
        loss_sum += loss.item() * batch_n
        n_targets += batch_n

    return loss_sum / n_targets


def train(config: dict) -> dict:
    """Run the whole training procedure for one config and return its metrics."""
    set_seed(config["seed"])
    train_config = config["train"]
    device = pick_device(train_config["device"])

    vocab, streams = load_corpus(config)
    chunks = {
        split: tuple(t.to(device) for t in chunk(streams[split], train_config["context_length"], vocab.pad_id))
        for split in SPLITS
    }

    model = build_model(config, len(vocab))
    embedding = init_embeddings(model, config, vocab, streams["train"])
    model = model.to(device)
    frozen_table = input_embedding(model).weight.detach().clone() if embedding["frozen"] else None
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )
    steps_per_epoch = len(chunks["train"][0]) // train_config["batch_size"]
    schedule = make_schedule(optimizer, train_config, train_config["max_epochs"] * steps_per_epoch)

    run_dir = REPO_ROOT / config["output"]["runs_dir"] / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"{config['run_name']}: {parameters['total']:,} parameters on {device}, {steps_per_epoch} steps per epoch")

    log = []
    best = {"epoch": 0, "val_loss": math.inf, "train_seconds": 0.0}
    train_seconds = 0.0

    for epoch in range(1, train_config["max_epochs"] + 1):
        synchronize(device)
        start = time.perf_counter()
        train_loss = train_epoch(model, *chunks["train"], optimizer, schedule, train_config, vocab.pad_id)
        synchronize(device)
        train_seconds += time.perf_counter() - start

        val_loss = evaluate(model, *chunks["validation"], train_config["batch_size"], vocab.pad_id)

        # train_seconds counts the training passes only, not validation.
        # The learning rate at the end of the epoch is logged too: with a decaying
        # schedule, late epochs train at a small fraction of the peak rate.
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_seconds": train_seconds,
                    "lr": schedule.get_last_lr()[0]})
        with open(run_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)
        print(f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}  {train_seconds:7.1f}s")

        if val_loss < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": val_loss, "train_seconds": train_seconds}
            checkpoint = {"model": model.state_dict(), "config": config, "vocab_size": len(vocab), **best}
            torch.save(checkpoint, run_dir / "best.pt")
        elif epoch - best["epoch"] >= train_config["early_stopping_patience"]:
            print(f"no improvement for {epoch - best['epoch']} epochs, stopping")
            break

    # Final numbers come from the best checkpoint, not from the last epoch.
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
    if frozen_table is not None:
        # Evidence that freezing worked: the table in the final model is, bit
        # for bit, the one it started with.
        embedding["unchanged_after_training"] = torch.equal(input_embedding(model).weight, frozen_table)
    results = {
        "run_name": config["run_name"],
        "model": config["model"],
        "vocab_size": len(vocab),
        "parameters": parameters,
        "embedding": embedding,
        "device": str(device),
        "torch": torch.__version__,
        "epochs_run": len(log),
        "best_epoch": best["epoch"],
        "train_seconds_to_best": best["train_seconds"],
        "train_seconds_total": train_seconds,
        "loss": {},
        "perplexity": {},
    }
    for split in ("validation", "test"):
        loss = evaluate(model, *chunks[split], train_config["batch_size"], vocab.pad_id)
        results["loss"][split] = loss
        results["perplexity"][split] = math.exp(loss)
        print(f"{split:>10}: loss {loss:.4f}  perplexity {math.exp(loss):.2f}")

    results_dir = REPO_ROOT / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{config['run_name']}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
