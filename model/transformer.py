import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import stochastic_depth
from functools import partial
from torch_geometric.utils import scatter
from torch_geometric.nn.aggr import Set2Set
from model.base import AttentionBase

ACTIVATION = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
}


class MLP(nn.Module):
    def __init__(self, embed_dim, ffn_dim, ffn_dropout, bias, activation):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.w2 = nn.Linear(ffn_dim, embed_dim, bias=bias)
        self.d1 = nn.Dropout(ffn_dropout)
        self.d2 = nn.Dropout(ffn_dropout)
        self.activation = ACTIVATION[activation]

    def forward(self, x):
        return self.d2(self.w2(self.d1(self.activation(self.w1(x)))))


class GLU(nn.Module):
    def __init__(
        self, embed_dim, ffn_dim, ffn_dropout, bias, activation, multiple_of=256
    ):
        super().__init__()
        ffn_dim = int(2 * ffn_dim / 3)
        ffn_dim = multiple_of * ((ffn_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.w2 = nn.Linear(ffn_dim, embed_dim, bias=bias)
        self.w3 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.d1 = nn.Dropout(ffn_dropout)
        self.d2 = nn.Dropout(ffn_dropout)
        self.activation = ACTIVATION[activation]

    def forward(self, x):
        return self.d2(self.w2(self.d1(self.activation(self.w1(x)) * self.w3(x))))


FFN = {
    "mlp": MLP,
    "glu": GLU,
}


class MultiHeadAttention(AttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, bias: bool):
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout, bias, batch_first=True
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        kwargs = {"need_weights": False}
        if attn_mask is not None:
            if len(attn_mask.shape) == 3:
                kwargs["attn_mask"] = attn_mask
            else:
                kwargs["key_padding_mask"] = attn_mask
        return self.attention(x, x, x, **kwargs)[0]


class TransformerLayer(nn.Module):
    def __init__(
        self,
        attention_class,
        embed_dim,
        ffn_dim,
        num_heads,
        attention_dropout,
        ffn_dropout,
        depth_dropout,
        bias,
        ffn,
        activation,
        norm_first,
        elementwise_affine,
    ):
        super().__init__()
        self.attention = attention_class(embed_dim, num_heads, attention_dropout, bias)
        self.ffn = FFN[ffn](embed_dim, ffn_dim, ffn_dropout, bias, activation)
        self.norm1 = nn.LayerNorm(
            embed_dim, elementwise_affine=elementwise_affine, bias=bias
        )
        self.norm2 = nn.LayerNorm(
            embed_dim, elementwise_affine=elementwise_affine, bias=bias
        )
        self.norm_first = norm_first
        self.stochastic_depth = lambda x, training: (
            stochastic_depth(x, depth_dropout, "batch", training)
            if depth_dropout > 0
            else x
        )

    def forward(self, x, attn_mask=None):
        if self.norm_first:
            x = x + self.stochastic_depth(
                self.attention(self.norm1(x), attn_mask), self.training
            )
            x = x + self.stochastic_depth(self.ffn(self.norm2(x)), self.training)
        else:
            x = self.norm1(
                x + self.stochastic_depth(self.attention(x, attn_mask), self.training)
            )
            x = self.norm2(x + self.stochastic_depth(self.ffn(x), self.training))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        attention_class,
        num_layers,
        embed_dim,
        ffn_dim,
        num_heads,
        attention_dropout,
        ffn_dropout,
        bias,
        ffn="mlp",
        activation="relu",
        norm_first=False,
        use_stochastic_depth=False,
        input_ln=False,
        elementwise_affine=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    attention_class,
                    embed_dim,
                    ffn_dim,
                    num_heads,
                    attention_dropout,
                    ffn_dropout,
                    (0.1 * (i + 1) / num_layers) if use_stochastic_depth else 0.0,
                    bias,
                    ffn,
                    activation,
                    norm_first,
                    elementwise_affine,
                )
                for i in range(num_layers)
            ]
        )
        if input_ln:
            self.input_ln = nn.LayerNorm(
                embed_dim,
                elementwise_affine=elementwise_affine,
                bias=bias,
            )

    def forward(self, x, mask=None):
        if hasattr(self, "input_ln"):
            x = self.input_ln(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Pooling(nn.Module):
    def __init__(self, pooling, embed_dim):
        super().__init__()
        if pooling in ["sum", "mean"]:
            self.pooling = partial(scatter, reduce=pooling)
        elif pooling == "set2set":
            self.pooling = Set2Set(embed_dim, processing_steps=6)
        elif pooling is None:
            self.pooling = None
        else:
            raise ValueError(f"Pooling {pooling} is not supported")

    def forward(self, x, batch):
        if self.pooling is not None:
            return self.pooling(x, batch)
        return x
