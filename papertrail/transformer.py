"""Decoder-only Transformer language model.

    token ids -> embedding * sqrt(d_model) + sinusoidal positions -> dropout
              -> n_layers x [masked self-attention, feed-forward] -> linear -> logits

Ported by hand from an earlier encoder-decoder implementation
(github.com/nabilrakaiza/transformer), keeping only what a decoder-only
model needs.

Architecture notes:
  - Post-LN: each sublayer is LayerNorm(x + dropout(sublayer(x))). There is no
    final LayerNorm after the stack (each block already ends normalized), and
    training needs learning-rate warmup.
  - Sinusoidal positions, ReLU, and the sqrt(d_model) embedding scale, as in
    Vaswani et al. (2017).
  - Weights initialized at normal(std=0.02), biases at zero, as GPT-1 does.
    This pass is mandatory, not cosmetic: the blocks are built with clones(),
    so without it every layer would start with identical weights.
  - The output layer is NOT tied to the input embedding, unlike GPT-1. No
    model in this project ties them (see "Design choices" in the README).
"""
import copy
import math

import torch
from torch import nn
from torch.nn.functional import softmax


# ---------------------------------------------------------------- helpers

def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Produce n deep copies of a module as a ModuleList.

    Note the copies share their *initial values*: deepcopy copies weights,
    not just structure. Whatever assembles these must run an init pass
    afterwards, or every layer starts identical.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# ------------------------------------------------------------- attention

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Leading dimensions are treated as batch dimensions by matmul, so any
    rank works. In practice MultiHeadAttention calls this with 4-D tensors,
    after the heads have been split out.

    query: (..., seq_len_q, d_k)   in practice (batch, n_heads, seq_len, d_k)
    key:   (..., seq_len_k, d_k)
    value: (..., seq_len_k, d_v)
    mask:  broadcastable to (..., seq_len_q, seq_len_k).
           True / 1 = attend, 0 = block. Note this is the OPPOSITE of
           PyTorch's own boolean attn_mask, where True means masked out.

    Returns the output, shape (..., seq_len_q, d_v), and the attention
    weights after dropout, shape (..., seq_len_q, seq_len_k).
    """
    d_k = query.size(-1)

    # scores[..., i, j] = dot(query_i, key_j): every query against every key.
    # Rows are queries, columns are keys, which is why the softmax and the
    # causal mask both act on the last dimension.
    scores = torch.matmul(query, key.transpose(-1, -2))

    # sqrt(d_k) keeps the dot products from growing with dimension, which
    # would otherwise push softmax into its saturated, low-gradient regime.
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        # finfo(dtype).min rather than -1e9: -1e9 overflows to -inf in fp16,
        # and a fully-masked row would then softmax to NaN.
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

    attention_weights = softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    return torch.matmul(attention_weights, value), attention_weights


class MultiHeadAttention(nn.Module):
    """n_heads attention heads run in parallel over disjoint slices of d_model.

    Splitting d_model into heads of d_k = d_model / n_heads is what makes
    multi-head free: the heads together cost exactly the same parameters and
    FLOPs as one full-width head. What you buy is n_heads independent
    attention patterns per token instead of one averaged compromise.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # d_v = d_k

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k).

        Moving the heads next to the batch dimension makes attention treat
        each head as an independent batch item.
        """
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query, key, value: (batch, seq_len, d_model)
        mask: broadcastable to (batch, n_heads, seq_len, seq_len).
              (seq_len, seq_len) is enough for a causal mask, since it is
              the same for every batch element and every head.
              True / 1 = attend, 0 = block.

        Returns a tensor of shape (batch, seq_len, d_model).
        """
        batch_size = query.size(0)

        q_heads = self.split_heads(self.w_q(query))
        k_heads = self.split_heads(self.w_k(key))
        v_heads = self.split_heads(self.w_v(value))

        attn_output, _ = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask, self.dropout)

        # Combine the heads back. .contiguous() is required because transpose
        # returns a strided view, and view() cannot flatten across
        # non-contiguous strides.
        concat_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # w_out lets the model mix the heads; without it each head would write
        # to a fixed, non-overlapping slice of the residual stream forever.
        return self.w_out(concat_output)


# ----------------------------------------------------------- feed-forward

class PositionWiseFeedForward(nn.Module):
    """Two linear layers with a ReLU, applied independently at every position.

    "Position-wise" is the whole point: this layer cannot move information
    between tokens. All cross-token mixing happens in attention. At the
    conventional d_ff = 4 * d_model, this holds two thirds of a block's
    parameters.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) in and out."""
        return self.w_2(self.dropout(self.w_1(x).relu()))


# ------------------------------------------------------------- embeddings

class Embeddings(nn.Module):
    """Token id -> dense vector, scaled by sqrt(d_model).

    The scaling comes from Vaswani section 3.4. It only lands in a sensible
    range if the embedding matrix is initialized small: with PyTorch's
    default N(0, 1) the token embeddings would be about sqrt(d_model) times
    larger than the positional signal. The init pass in Transformer
    (std=0.02) is what makes it work.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)  # lut is lookup table
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len) of token ids -> (batch, seq_len, d_model)."""
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position signal, added to the token embeddings.

    Each dimension is a sinusoid at a different frequency, which makes
    PE(pos + k) a fixed linear transform of PE(pos) for any offset k. That
    is what lets attention learn relative-position behaviour from absolute
    encodings.

    Nothing here is learned.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        # sin fills the even columns, cos the odd ones, so they must be equal in number
        assert d_model % 2 == 0, "d_model must be even"
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # exp(-2i * log(10000) / d) instead of 1 / 10000**(2i/d), which loses
        # precision at high frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # A buffer so it follows the model to the GPU. persistent=False: pe is
        # a fixed function of position, so recomputing it on load is free and
        # keeps it out of checkpoints.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) in and out: adds position, then dropout."""
        return self.dropout(x + self.pe[:, : x.size(1)])


# ------------------------------------------------ normalization / residual

class LayerNorm(nn.Module):
    """Layer normalization with a learnable gain (a_2) and bias (b_2).

    Uses the biased variance with eps inside the sqrt, matching nn.LayerNorm.
    (torch.std would apply Bessel's n-1 correction, and eps outside the sqrt
    under-damps near-constant activations by ~900x.)
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes over the last dimension. Shape is unchanged."""
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Residual connection around a sublayer, post-LN.

        out = LayerNorm(x + dropout(sublayer(x)))

    This is the ordering of the original Transformer and GPT-1: normalize
    *after* the residual add. GPT-2 moved the norm to the sublayer input
    (pre-LN). Two consequences of post-LN: no final LayerNorm is needed after
    the block stack, and training needs learning-rate warmup.

    Note dropout wraps only the sublayer output, never the residual: the
    identity path has to stay clean for gradients to reach early layers.
    """

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """
        x:        (batch, seq_len, d_model), from the previous layer
        sublayer: a function or module mapping x to a tensor of the same shape

        Returns a tensor of the same shape as x.
        """
        return self.norm(x + self.dropout(sublayer(x)))


# ------------------------------------------------------------------ blocks

class DecoderLayer(nn.Module):
    """One post-LN decoder block: masked self-attention, then feed-forward.

    Two sublayers, not three: there is no encoder to cross-attend to, which
    is the whole difference between this and the original encoder-decoder
    DecoderLayer.
    """

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadAttention,
        feed_forward: PositionWiseFeedForward,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) in and out."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """A stack of n_layers identical DecoderLayers.

    No final LayerNorm: that belongs to pre-LN designs, where nothing else
    ever normalizes the residual stream. GPT-1 has none; GPT-2 added one
    exactly when it moved to pre-LN.
    """

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        layer = DecoderLayer(
            d_model,
            MultiHeadAttention(d_model, n_heads, dropout),
            PositionWiseFeedForward(d_model, d_ff, dropout),
            dropout,
        )
        self.layers = clones(layer, n_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) in and out."""
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ------------------------------------------------------------------- model

class Transformer(nn.Module):
    """Decoder-only Transformer language model.

    vocab_size: vocabulary size
    n_layers:   number of decoder blocks
    d_model:    residual stream width
    n_heads:    number of attention heads (must divide d_model)
    d_ff:       feed-forward hidden width, conventionally 4 * d_model
    dropout:    applied to attention weights, the FFN hidden layer, sublayer
                outputs, and the embedding sum
    max_len:    maximum sequence length. Sizes the positional encoding and
                the causal mask, so mask memory grows quadratically in it.
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.embeddings = Embeddings(d_model, vocab_size)
        self.positional_embeddings = PositionalEncoding(d_model, dropout, max_len)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.head = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

        # Cached causal mask: position i may attend to positions j <= i.
        # A buffer so it follows the model to the GPU; persistent=False so it
        # stays out of checkpoints. Slicing works because a causal mask is
        # nested: the top-left T x T corner of a bigger one is exactly the
        # T x T mask.
        i = torch.arange(max_len)
        self.register_buffer("causal_mask", i[:, None] >= i[None, :], persistent=False)

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-1's initialization: normal(std=0.02) weights, zero biases.

        Called once per submodule by nn.Module.apply, which handles the
        recursion. LayerNorm is deliberately skipped: its gain and bias are
        already correct at ones and zeros.
        """
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)

    def features(self, inputs: torch.Tensor, state: None = None) -> tuple[torch.Tensor, None]:
        """Final-layer hidden state at every position.

        inputs: (batch, seq_len) token ids.
        Returns the hidden states, shape (batch, seq_len, d_model), and None.
        Part III builds its sentence representations from these hidden states.
        """
        seq_len = inputs.size(1)
        max_len = self.causal_mask.size(0)
        assert seq_len <= max_len, f"sequence length {seq_len} exceeds max_len {max_len}"

        x = self.positional_embeddings(self.embeddings(inputs))
        return self.decoder(x, self.causal_mask[:seq_len, :seq_len]), None

    def forward(self, inputs: torch.Tensor, state: None = None) -> tuple[torch.Tensor, None]:
        """Next-token logits at every position.

        inputs: (batch, seq_len) token ids.
        state:  unused. A Transformer carries nothing from one chunk to the
                next; the argument and the second return value exist so that
                every model in the project is called the same way.
        Returns logits of shape (batch, seq_len, vocab_size), and None.
        """
        hidden, _ = self.features(inputs)
        return self.head(hidden), None
