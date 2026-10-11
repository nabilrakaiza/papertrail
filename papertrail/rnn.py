"""Recurrent language models: Elman RNN and LSTM.

The two differ only in the recurrent layer, so one class covers both:

    embedding -> dropout -> [RNN or LSTM layer -> dropout] x num_layers -> linear -> logits

A recurrent layer reads left to right, so the state at position t depends only
on tokens 0..t. That is what makes the model causal; no mask is needed.

The layers are stacked by hand, one single-layer module each, with an ordinary
nn.Dropout between them. PyTorch can do the stacking itself
(`nn.LSTM(num_layers=2, dropout=...)`), but in PyTorch 2.9 that built-in
dropout is broken for the LSTM on Apple's `mps` device: whichever mode the
first call in the process uses, training or evaluation, is silently used for
every later call. Stacking by hand computes the same thing and avoids it.
"""
import torch
from torch import nn

# What one recurrent layer carries from one step to the next: a hidden tensor
# for the RNN, a (hidden, cell) pair for the LSTM, each tensor of shape
# (1, batch, hidden_dim). The model's state is a list with one entry per layer.
LayerState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]
State = list[LayerState]


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

        if cell == "rnn":
            make_layer = lambda input_dim: nn.RNN(input_dim, hidden_dim, nonlinearity="tanh", batch_first=True)
        elif cell == "lstm":
            make_layer = lambda input_dim: nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"unknown cell: {cell!r}")
        # The first layer reads embeddings, every later one reads the layer below.
        self.layers = nn.ModuleList(
            make_layer(emb_dim if i == 0 else hidden_dim) for i in range(num_layers)
        )

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
        if state is None:
            state = [None] * len(self.layers)

        hidden = self.embedding(inputs)
        new_state = []
        for layer, layer_state in zip(self.layers, state):
            # Dropout on what each layer reads: the embeddings for the first
            # layer, the output of the layer below for the others.
            hidden, layer_state = layer(self.dropout(hidden), layer_state)
            new_state.append(layer_state)
        return hidden, new_state

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


def detach(state):
    """Keep the values of a state but cut it off from the graph that produced it.

    Truncated backpropagation through time: the next chunk starts from this
    state, but its gradients stop here and do not flow back into earlier chunks.
    """
    if isinstance(state, (list, tuple)):
        return type(state)(detach(s) for s in state)
    return state.detach()
