"""Build a neural language model from a config, and count its parameters."""
from torch import nn

from papertrail.rnn import RecurrentLM
from papertrail.transformer import Transformer


def build_model(config: dict, vocab_size: int) -> nn.Module:
    """Create the model described by `config["model"]`, with fresh random weights.

    Every model built here is called the same way:

        logits, state = model(inputs, state)      # logits: (batch, time, vocab_size)
        hidden, state = model.features(inputs)    # hidden: (batch, time, width)
    """
    model = config["model"]

    # Refuse settings this function does not act on, rather than ignore them.
    if model["tie_weights"]:
        raise NotImplementedError("tie_weights: no model in this project ties its embeddings")
    if model["emb_init"] != "random" or model["freeze_emb"]:
        raise NotImplementedError("pretrained and frozen embeddings (Part II) are not written yet")

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
