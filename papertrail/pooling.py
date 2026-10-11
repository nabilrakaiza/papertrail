"""Turn a language model's per-token hidden states into one vector per text.

A probe needs a fixed-length representation of each news item, but a language
model produces one hidden state per token. These are the two ways of collapsing
them that Part III compares.

Both functions take a whole batch at once:

    hidden:  (batch, time, width)  hidden state at every position, from model.features(ids)
    lengths: (batch,)              number of real tokens in each row; the rest is <pad>

and return a tensor of shape (batch, width).
"""
import torch


def mean_pool(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """The average hidden state over the real tokens of each row.

    Padded positions must not count, neither in the sum nor in the divisor.
    """
    time = hidden.size(1)

    # A table of shape (batch, time): True where a position holds a real token.
    # Position numbers, shaped (1, time), are compared with each row's length,
    # shaped (batch, 1); broadcasting fills in every combination.
    #
    #     lengths = [3, 1]      positions:  0  1  2
    #     row 0                             T  T  T
    #     row 1                             T  F  F
    positions = torch.arange(time, device=hidden.device)
    real = positions[None, :] < lengths[:, None]

    # Multiplying by the table turns padded positions into zeros, so they add
    # nothing to the sum. The table needs a third dimension of size 1 to line
    # up with (batch, time, width).
    total = (hidden * real[:, :, None]).sum(dim=1)

    # Divide each row by its own number of real tokens, not by `time`.
    return total / lengths[:, None]


def last_pool(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """The hidden state at the last real token of each row.

    That is position `length - 1`, which differs from row to row. It is not
    the final column: for a short text the final column is padding.

    Both language models read left to right, so this one state has seen the
    whole text.
    """
    # Indexing with two tensors of the same length picks one element per pair:
    # (row 0, its last position), (row 1, its last position), and so on.
    rows = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[rows, lengths - 1]
