"""Recurrent language models: Elman RNN and LSTM.

The two differ only in the recurrent layer, so one class covers both:

    embedding -> dropout -> RNN or LSTM (stacked) -> dropout -> linear -> logits

A recurrent layer reads left to right, so the state at position t depends only
on tokens 0..t. That is what makes the model causal; no mask is needed.
"""
import torch
from torch import nn

# What the recurrent layer carries from one step to the next: a hidden tensor
# for the RNN, a (hidden, cell) pair for the LSTM. Each tensor has shape
# (num_layers, batch, hidden_dim).
State = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class RecurrentLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        cell: str,
    ):
        """`cell` is "rnn" (Elman, tanh) or "lstm"."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # `dropout` here acts between the stacked layers, not after the last one.
        if cell == "rnn":
            self.recurrent = nn.RNN(
                emb_dim, hidden_dim, num_layers, nonlinearity="tanh", dropout=dropout, batch_first=True
            )
        elif cell == "lstm":
            self.recurrent = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        else:
            raise ValueError(f"unknown cell: {cell!r}")

        self.dropout = nn.Dropout(dropout)
        # A separate output matrix: input and output embeddings are not tied
        # (see "Design choices" in the README).
        self.output = nn.Linear(hidden_dim, vocab_size)

    def features(self, inputs: torch.Tensor, state: State | None = None) -> tuple[torch.Tensor, State]:
        """Top-layer hidden state at every position.

        inputs: (batch, time) token ids.
        Returns the hidden states, shape (batch, time, hidden_dim), and the
        state after the last position. Part III builds its sentence
        representations from these hidden states.
        """
        embedded = self.dropout(self.embedding(inputs))
        return self.recurrent(embedded, state)

    def forward(self, inputs: torch.Tensor, state: State | None = None) -> tuple[torch.Tensor, State]:
        """Next-token logits at every position.

        inputs: (batch, time) token ids.
        state:  where to start from. None means zeros, i.e. no memory of
                anything before this chunk.
        Returns logits of shape (batch, time, vocab_size), and the state after
        the last position, which can be passed back in with the next chunk.
        """
        hidden, state = self.features(inputs, state)
        return self.output(self.dropout(hidden)), state


def detach(state: State) -> State:
    """Keep the values of a state but cut it off from the graph that produced it.

    Truncated backpropagation through time: the next chunk starts from this
    state, but its gradients stop here and do not flow back into earlier chunks.
    """
    if isinstance(state, tuple):
        return tuple(s.detach() for s in state)
    return state.detach()
