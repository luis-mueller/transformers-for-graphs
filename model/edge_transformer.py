import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj, scatter
from model.base import ModelBase, AttentionBase
from model.transformer import Transformer, Pooling


class FeatureEncoder(torch.nn.Module):
    def __init__(
        self,
        node_encoder,
        edge_encoder,
    ):
        super().__init__()
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder

    def forward(self, data):
        if not hasattr(data, "x") or data.x is None:
            data.x = torch.zeros(
                (data.num_nodes,), device=data.edge_index.device, dtype=torch.long
            )
        data.x = self.node_encoder(data.x)

        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            data.edge_attr = torch.zeros_like(data.edge_index[0]).to(torch.long)

        data.edge_attr = self.edge_encoder(data.edge_attr)
        return data


class MLP(torch.nn.Sequential):
    def __init__(
        self, input_dim, output_dim, dropout: float = 0.0, linear: bool = False
    ):
        if not linear:
            hidden_dim = output_dim

            layers = [
                torch.nn.BatchNorm1d(input_dim),
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim),
                torch.nn.Dropout(dropout),
            ]
            super().__init__(*layers)
        else:
            super().__init__(
                torch.nn.Linear(input_dim, output_dim),
            )


class Composer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        linear: bool = True,
    ):
        super().__init__()
        concat_dim = 2 * embed_dim
        self.node_proj = MLP(concat_dim, embed_dim, linear=linear)

    def forward(self, x, edge_index, edge_attr, batch, token_index, token_attr=None):
        edge_features = to_dense_adj(edge_index, batch, edge_attr)

        if token_attr is not None:
            token_attr = to_dense_adj(token_index, batch, token_attr)
            edge_features = torch.cat([edge_features, token_attr], -1)

        x = x[token_index.T].flatten(1, 2)
        x = self.node_proj(x)
        x = to_dense_adj(token_index, batch, x)
        x = x + edge_features
        return x


class Decomposer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        reduce_fn="sum",
    ):
        super().__init__()
        self.node_dim = embed_dim
        self.reduce_fn = reduce_fn

        self.out_proj = MLP(embed_dim, 2 * embed_dim)
        self.node_mlp = MLP(self.node_dim, embed_dim)

    def forward(self, x, node_features, node_batch, token_index):
        x = self.out_proj(x)

        dim_size = node_batch.size(0)
        node_features = torch.zeros_like(node_features)

        for i in range(2):
            features_order_i = x[:, i * self.node_dim : (i + 1) * self.node_dim]
            features_order_i = scatter(
                features_order_i,
                token_index[i],
                0,
                dim_size=dim_size,
                reduce=self.reduce_fn,
            )
            node_features = node_features + features_order_i

        return self.node_mlp(node_features)


def apply_mask_2d(node_features, node_batch):
    _, mask = to_dense_batch(node_features, node_batch)
    unbatch = mask.unsqueeze(2) * mask.unsqueeze(1)  # B x N x N
    mask = unbatch.unsqueeze(3) * mask.unsqueeze(1).unsqueeze(2)  # B x N x N x N
    return unbatch, mask


def triang_attn(q, k):
    out = q.unsqueeze(3) * k.unsqueeze(1)
    return out.sum(dim=5)


def val_fusion(v1, v2):
    return v1.unsqueeze(3) * v2.unsqueeze(1)


def final_comp(att, val):
    out = att.unsqueeze(-1) * val
    return out.sum(dim=2)


class EdgeAttention(AttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, bias: bool):
        super().__init__(embed_dim, num_heads, dropout, bias)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.qlin = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.klin = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v1lin = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v2lin = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.olin = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = torch.nn.Dropout(p=dropout)

    @torch.compile
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        query = x
        key = x
        value = x
        mask = attn_mask
        num_batches = query.size(0)
        num_nodes_q = query.size(1)
        num_nodes_k = key.size(1)

        left_k = self.qlin(query)
        right_k = self.klin(key)
        left_v = self.v1lin(value)
        right_v = self.v2lin(value)

        left_k = left_k.view(
            num_batches, num_nodes_q, num_nodes_q, self.num_heads, self.d_k
        )
        right_k = right_k.view(
            num_batches, key.size(1), key.size(2), self.num_heads, self.d_k
        )
        left_v = left_v.view_as(right_k)
        right_v = right_v.view_as(right_k)

        if hasattr(self, "norms"):
            left_k = self.norms[0](left_k)
            right_k = self.norms[1](right_k)

        scores = triang_attn(left_k, right_k) / math.sqrt(self.d_k)

        if mask is not None:
            scores_dtype = scores.dtype
            scores = (
                scores.to(torch.float32)
                .masked_fill(mask.unsqueeze(4), -1e9)
                .to(scores_dtype)
            )

        att = F.softmax(scores, dim=2)
        att = self.dropout(att)
        val = val_fusion(left_v, right_v)

        if hasattr(self, "norms"):
            val = self.norms[2](val)

        x = final_comp(att, val)
        x = x.view(num_batches, num_nodes_q, num_nodes_k, self.embed_dim)
        return self.olin(x)


class EdgeTransformer(ModelBase):
    def __init__(
        self,
        modules: dict,
        pe_encoder: nn.Module,
        num_layers: int,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        bias: bool,
        ffn: str,
        activation: str,
        norm_first: bool,
        pooling: str,
    ):
        super().__init__(
            modules,
            pe_encoder,
            num_layers,
            embed_dim,
            ffn_dim,
            num_heads,
            attention_dropout,
            ffn_dropout,
            bias,
            ffn,
            activation,
            norm_first,
            pooling,
        )

        if modules["node"] is None:
            modules["node"] = torch.nn.Embedding(1, embed_dim)
        if modules["edge"] is None:
            modules["edge"] = torch.nn.Embedding(1, embed_dim)

        self.feature_encoder = FeatureEncoder(
            modules["node"],
            modules["edge"],
        )
        self.composer = Composer(embed_dim)

        self.transformer = Transformer(
            EdgeAttention,
            num_layers,
            embed_dim,
            ffn_dim,
            num_heads,
            attention_dropout,
            ffn_dropout,
            bias,
            ffn,
            activation,
            norm_first,
            use_stochastic_depth=True,
            input_ln=True,
            elementwise_affine=False,
        )
        self.decomposer = Decomposer(embed_dim)

        self.pooling = Pooling(pooling, embed_dim)

    def forward(self, data, return_graph=False, return_node=False, return_edge=False):
        data = self.feature_encoder(data)
        token_attr = data.token_attr if hasattr(data, "token_attr") else None
        token_index = data.token_index

        x = self.composer(
            data.x, data.edge_index, data.edge_attr, data.batch, token_index, token_attr
        )
        unbatch, mask = apply_mask_2d(data.x, data.batch)
        x = self.transformer(x, ~mask)

        if return_graph:
            x = x[unbatch]
            x = self.decomposer(x, data.x, data.batch, token_index)
            return self.pooling(x, data.batch)
        elif return_node:
            x = x[unbatch]
            x = self.decomposer(x, data.x, data.batch, token_index)
            return x
        elif return_edge:
            edge_embeds = []
            for i, _data in enumerate(data.to_data_list()):
                edge_embeds.append(x[i, _data.edge_index[0], _data.edge_index[1]])
            return torch.cat(edge_embeds)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
