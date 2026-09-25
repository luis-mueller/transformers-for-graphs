import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch, to_dense_adj
from model.base import ModelBase
from model.transformer import Transformer, MultiHeadAttention, Pooling
from model.utils import add_to_attn_mask


class CLSToken(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn((1, 1, embed_dim)))
        self.cls_loop = nn.Parameter(torch.randn((1, 1, 1, num_heads)))
        self.cls_in = nn.Parameter(torch.randn((1, 1, 1, num_heads)))
        self.cls_out = nn.Parameter(torch.randn((1, 1, 1, num_heads)))

    def forward(self, x, attn_mask):
        # B, N, D
        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], 1)

        # B, 1, (N - 1), H
        attn_col = self.cls_out.repeat(x.size(0), 1, x.size(1) - 1, 1)

        # B, (N - 1), 1, H
        attn_row = torch.cat(
            [self.cls_loop, self.cls_in.repeat(1, x.size(1) - 1, 1, 1)], 1
        )
        attn_row = attn_row.repeat(x.size(0), 1, 1, 1)

        attn_mask = torch.cat([attn_col, attn_mask], 1)  # B, N, (N - 1), H
        attn_mask = torch.cat([attn_row, attn_mask], 2)  # B, N, N, H

        return x, attn_mask


class TokenEmbedding(nn.Module):
    def __init__(self, modules, pe_encoder, embed_dim, num_heads):
        super().__init__()
        self.embed_modules = modules
        self.pe_encoder = pe_encoder
        self.edge_encoder = nn.Linear(embed_dim, num_heads)
        self.node_embedding = nn.Parameter(torch.randn((1, embed_dim)))
        self.edge_embedding = nn.Parameter(torch.randn((1, embed_dim)))
        self.edge_transform_edge_embedding = nn.Embedding(3, embed_dim)
        self.cls_token = CLSToken(embed_dim, num_heads)

    def embed_nodes(self, loc, data, num_nodes):
        if self.embed_modules["node"] is not None:
            if "node_depth" in data and data["node_depth"] is not None:
                return self.embed_modules["node"](data[loc], data.node_depth)
            return self.embed_modules["node"](data[loc])

        if loc in data and data[loc] is not None:
            return self.node_embedding.repeat(data[loc].size(0), 1)
        return self.node_embedding.repeat(num_nodes, 1)

    def embed_edges(self, edge_encoder, loc, data, num_edges):
        if edge_encoder is not None and loc in data and data[loc] is not None:
            return edge_encoder(data[loc])
        return self.edge_embedding.repeat(num_edges, 1)

    def forward(self, data):
        if hasattr(data, "token_mask"):
            x = torch.zeros(
                (data.token_mask.size(0), self.node_embedding.size(1)),
                dtype=self.node_embedding.dtype,
                device=self.node_embedding.device,
            )
            if (num_node_tokens := torch.sum(~data.token_mask)) > 0:
                x[~data.token_mask] = self.embed_nodes(
                    "x_node", data, num_node_tokens
                ).to(x.dtype)

            x[data.token_mask] = self.embed_edges(
                self.embed_modules["edge"],
                "x_edge",
                data,
                data.num_nodes - num_node_tokens,
            ).to(x.dtype)
            edge_attr = self.embed_edges(
                self.edge_transform_edge_embedding,
                "edge_attr",
                data,
                data.edge_index.size(1),
            )
        else:
            x = self.embed_nodes("x", data, data.num_nodes)
            edge_attr = self.embed_edges(
                self.embed_modules["edge"],
                "edge_attr",
                data,
                data.edge_index.size(1),
            )

        absolute_pe, relative_pe = self.pe_encoder(data, self.training)
        if absolute_pe is not None:
            x = x + absolute_pe

        x, mask = to_dense_batch(x, data.batch)

        # NOTE: Allows to disable the NoPE relative embedding fallback
        if edge_attr is not None:
            e = self.edge_encoder(edge_attr)
            attn_mask = to_dense_adj(data.edge_index, data.batch, e)
        else:
            attn_mask = torch.zeros(
                (x.size(0), x.size(1), x.size(1), self.num_heads),
                dtype=x.dtype,
                device=x.device,
            )

        if relative_pe is not None:
            add_to_attn_mask(attn_mask, relative_pe, data.sparse_pos, data.batch)

        attn_mask[~mask] = float("-inf")
        attn_mask.permute(0, 2, 1, 3)[~mask] = float("-inf")
        x, attn_mask = self.cls_token(x, attn_mask)
        attn_mask = attn_mask.permute(0, 3, 1, 2).flatten(0, 1)

        return x, mask, attn_mask


class GDTransformer(ModelBase):
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

        self.token_embed = TokenEmbedding(
            modules,
            pe_encoder,
            embed_dim,
            num_heads,
        )

        self.transformer = Transformer(
            MultiHeadAttention,
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

    def forward(self, data, return_graph=False, return_node=False, return_edge=False):
        x, mask, attn_mask = self.token_embed(data)
        x = self.transformer(x, attn_mask)

        if return_graph:
            return x[:, 0]
        elif return_node:
            if hasattr(data, "token_mask"):
                return x[:, 1:][mask][~data.token_mask]
            return x[:, 1:][mask]
        elif return_edge:
            return x[:, 1:][mask][data.token_mask]

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
