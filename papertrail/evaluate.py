"""Evaluation shared by every neural language model.

One function, used for the validation loss after each epoch and for the final
validation and test numbers, so that all models are scored the same way.
"""
import torch
from torch.nn import functional as F


def evaluate(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    pad_id: int,
) -> float:
    """Mean cross-entropy (in nats) over every non-pad target.

    inputs, targets: (n_chunks, context_length) token ids, from `chunk` in
                     papertrail/data.py, already on the model's device.
    Perplexity is exp of the returned value.

    Every chunk starts from a fresh state, so recurrent models see at most
    `context_length` tokens of history, the same as the Transformer.
    """
    # Dropout off. The caller is responsible for switching back to train mode.
    model.eval()

    total = count = 0

    # No gradient tracking: nothing is trained here. Unlike setting
    # requires_grad, this ends when the block does.
    with torch.no_grad():
        # Every chunk, in order. Slicing past the end just gives a shorter
        # slice, so the last, smaller batch needs no special case.
        for i in range(0, len(targets), batch_size):
            inp = inputs[i:i+batch_size]
            tar = targets[i:i+batch_size]

            # No state is passed, so each chunk starts fresh.
            logits, _ = model(inp)

            # A sum, not a mean: batches differ in size and the final chunk
            # is padded, so only one division at the very end gives the true
            # mean per target.
            total += F.cross_entropy(
                logits.flatten(0, 1), tar.flatten(), ignore_index=pad_id, reduction="sum"
            ).item()
            count += (tar != pad_id).sum().item()

    return total / count
