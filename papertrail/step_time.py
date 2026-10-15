"""Training cost per epoch for every neural model, measured under the same conditions.

    python -m papertrail.step_time --configs configs/part1/rnn.yaml configs/part1/lstm.yaml \
        configs/part1/transformer.yaml configs/part2/lstm_w2v_frozen.yaml configs/part2/lstm_glove_frozen.yaml

The wall-clock times recorded during the full training runs are not comparable
with each other: the runs happened at different times, on a laptop whose speed
depends on its power state. So each model is timed again here, back to back,
for a short burst of real training steps (the same step as in train.py: forward
pass, loss, backward pass, gradient clipping, optimizer step). Time per epoch is
the median step time times the number of steps in an epoch. The models are
measured in turn, `--rounds` times over, and averaged. Writes the file named by
--output, which also records the power state the numbers belong to.
"""
import argparse
import gc
import json
import subprocess
import time

import torch
from torch.nn import functional as F

from papertrail.config import REPO_ROOT, load_config
from papertrail.data import chunk, load_corpus, sequential_batches
from papertrail.embeddings import init_embeddings
from papertrail.models import build_model
from papertrail.rnn import detach
from papertrail.utils import pick_device, set_seed, synchronize

N_BATCHES = 30   # steps timed per measurement
SKIP = 8         # the first steps pay one-off costs (memory allocation, kernel compilation)


def power_state() -> str:
    """The laptop's power source and Low Power Mode setting (macOS only)."""
    try:
        source = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0]
        low_power = [line.split()[-1] for line in subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout.splitlines()
                     if "lowpowermode" in line]
        return f"{source.split(chr(39))[1] if chr(39) in source else source}, Low Power Mode {'on' if low_power == ['1'] else 'off'}"
    except (OSError, IndexError):
        return "unknown"


def time_steps(config: dict, vocab, streams: dict, device: torch.device) -> float:
    """Median seconds per training step for one config, from a freshly built model."""
    train_config = config["train"]
    set_seed(config["seed"])
    model = build_model(config, len(vocab))
    init_embeddings(model, config, vocab, streams["train"])
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=train_config["lr"], weight_decay=train_config["weight_decay"]
    )
    tokens = N_BATCHES * train_config["batch_size"] * train_config["context_length"] + 1
    inputs, targets = (t.to(device) for t in chunk(streams["train"][:tokens], train_config["context_length"], vocab.pad_id))

    state, step_times = None, []
    for batch_inputs, batch_targets in sequential_batches(inputs, targets, train_config["batch_size"]):
        synchronize(device)
        start = time.perf_counter()
        logits, state = model(batch_inputs, state)
        loss = F.cross_entropy(logits.flatten(0, 1), batch_targets.flatten(), ignore_index=vocab.pad_id)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
        optimizer.step()
        state = detach(state) if state is not None and train_config["carry_hidden_train"] else None
        loss.item()
        synchronize(device)
        step_times.append(time.perf_counter() - start)

    del model, optimizer, state, inputs, targets
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    measured = sorted(step_times[SKIP:])
    return measured[len(measured) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", default="results/step_times.json")
    args = parser.parse_args()

    configs = [load_config(path) for path in args.configs]
    vocab, streams = load_corpus(configs[0])
    device = pick_device(configs[0]["train"]["device"])
    steps_per_epoch = len(chunk(streams["train"], configs[0]["train"]["context_length"], vocab.pad_id)[0]) // configs[0]["train"]["batch_size"]
    state = power_state()
    print(f"{state}; {device}; {steps_per_epoch} steps per epoch")

    per_round = {config["run_name"]: [] for config in configs}
    for round_number in range(1, args.rounds + 1):
        for config in configs:
            seconds = time_steps(config, vocab, streams, device)
            per_round[config["run_name"]].append(seconds)
            print(f"round {round_number}  {config['run_name']:<18} {seconds * 1000:6.0f} ms per step")

    results = {
        "power_state": state,
        "device": str(device),
        "torch": torch.__version__,
        "steps_per_epoch": steps_per_epoch,
        "rounds": args.rounds,
        "runs": {
            name: {
                "seconds_per_step_by_round": values,
                "seconds_per_epoch": sum(values) / len(values) * steps_per_epoch,
            }
            for name, values in per_round.items()
        },
    }
    for name, run in results["runs"].items():
        print(f"{name:<18} {run['seconds_per_epoch']:5.0f} s per epoch")
    with open(REPO_ROOT / args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
