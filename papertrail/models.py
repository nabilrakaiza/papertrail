"""Build a neural language model from a config or a checkpoint, and count its parameters."""
from pathlib import Path

import torch
from torch import nn

from papertrail.rnn import RecurrentLM
from papertrail.transformer import Transformer


def build_model(config: dict, vocab_size: int) -> nn.Module:
    """Create the model described by `config["model"]`, with fresh random weights.

    Every model built here is called the same way:

        logits, state = model(inputs, state)      # logits: (batch, time, vocab_size)
        hidden, state = model.features(inputs)    # hidden: (batch, time, width)

    The input embeddings always start random here. For the Part II runs,
    `init_embeddings` in papertrail/embeddings.py then replaces and freezes them.
    """
    model = config["model"]

    # Refuse a setting this function does not act on, rather than ignore it.
    if model["tie_weights"]:
        raise NotImplementedError("tie_weights: no model in this project ties its embeddings")

    if model["type"] in ("rnn", "lstm"):
        return RecurrentLM(
            vocab_size,
            model["emb_dim"],
            model["hidden_dim"],
            model["num_layers"],
            model["dropout"],
            cell=model["type"],
        )
    if model["type"] == "transformer":
        # The token embedding is the residual stream, so the two widths are one number.
        assert model["d_model"] == model["emb_dim"], "d_model must equal emb_dim"
        return Transformer(
            vocab_size,
            model["n_layers"],
            model["d_model"],
            model["n_heads"],
            model["d_ff"],
            model["dropout"],
            model["max_len"],
        )
    raise ValueError(f"unknown model type: {model['type']!r}")


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild a model saved by the training loop, with its trained weights.

    Returns the model, in evaluation mode, and the checkpoint dict, which also
    holds the run's config, vocabulary size, best epoch and validation loss.
    """
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["config"], checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Parameter counts, with the input embedding table reported separately.

    The embedding table is a large share of every model here and is the part
    Part II swaps out, so the comparison needs it as its own number.
    """
    embedding = sum(
        p.numel() for module in model.modules() if isinstance(module, nn.Embedding) for p in module.parameters()
    )
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "embedding": embedding,
    }
