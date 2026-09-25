import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from model.base import ModelBase
from model.transformer import Transformer, MultiHeadAttention, Pooling
from model.utils import get_context_size


class PairEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.encoder = nn.Linear(2 * in_dim, out_dim, bias=bias)

    def forward(self, x, token_index):
        return self.encoder(torch.cat([x[token_index[0]], x[token_index[1]]], -1))


class ConcatEncoder(nn.Module):
    def __init__(self, num_in, in_dim, out_dim, bias=True):
        super().__init__()
        self.encoder = nn.Linear(num_in * in_dim, out_dim, bias=bias)

    def forward(self, x, token_index):
        return self.encoder(
            torch.cat([x[token_index[i]] for i in range(token_index.size(0))], -1)
        )


class PairTokenizer(nn.Module):
    def __init__(
        self,
        pe_encoder,
        atom_encoder,
        bond_encoder,
        embed_dim,
        bias,
        max_degree=1000,
        degree_dim=16,
    ):
        super().__init__()
        self.eigen_embed = pe_encoder
        self.eigen_encoder = PairEncoder(embed_dim, embed_dim, bias)

        if degree_dim > 0:
            self.degree_embed = nn.Embedding(max_degree, degree_dim)
            self.degree_encoder = PairEncoder(degree_dim, embed_dim, bias)
        if atom_encoder is None:
            atom_encoder = nn.Embedding(1, embed_dim)
        self.atom_encoder = atom_encoder
        self.node_encoder = PairEncoder(embed_dim, embed_dim, bias)
        if bond_encoder is None:
            bond_encoder = nn.Embedding(1, embed_dim)
        self.bond_encoder = bond_encoder
        self.graph_token = nn.Parameter(torch.zeros((1, 1, embed_dim)))
        self.embed_dim = embed_dim

    def forward(self, data):
        token_index, real_edges = data.token_index, data.real_edges

        assert torch.allclose(token_index[:, real_edges], data.edge_index)

        if not hasattr(data, "x") or data.x is None:
            data.x = torch.zeros(
                (data.num_nodes,), device=data.token_index.device, dtype=torch.long
            )

        x = self.atom_encoder(data.x)

        if hasattr(self, "degree_embed"):
            d = self.degree_embed(data.degrees)
        l, _ = self.eigen_embed(data, self.training)

        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            data.edge_attr = torch.zeros(
                (data.edge_index.size(1),),
                device=self.graph_token.device,
                dtype=torch.long,
            )

        e_embed = self.bond_encoder(data.edge_attr)
        e = torch.zeros(
            (token_index.size(1), x.size(1)), device=e_embed.device, dtype=e_embed.dtype
        )
        e[real_edges] = e_embed

        token_embed = (
            e + self.eigen_encoder(l, token_index) + self.node_encoder(x, token_index)
        )

        if hasattr(self, "degree_encoder"):
            token_embed = token_embed + self.degree_encoder(d, token_index)

        if not hasattr(data, "batch") or data.batch is None:
            batch = torch.zeros(data.x.size(0), device=data.x.device, dtype=torch.long)[
                token_index[0]
            ]
        else:
            batch = data.batch[token_index[0]]

        # NOTE: Account for the graph_token
        # context_size = get_context_size(batch) + 4 - 1
        token_embed, mask = to_dense_batch(
            token_embed, batch
        )  # , max_num_nodes=context_size)

        graph_token = self.graph_token.repeat(token_embed.size(0), 1, 1)
        token_embed = torch.cat([graph_token, token_embed], 1)

        padding_mask = torch.zeros(
            (mask.size(0), mask.size(1) + 1), dtype=mask.dtype, device=mask.device
        )
        padding_mask[:, 0] = 1
        padding_mask[:, 1:] = mask

        return token_embed, padding_mask


def distribute_edge_features(edge_index, edge_attr, token_index):
    order = token_index.size(0)
    distributed_edge_features = torch.zeros(
        (token_index.size(1), edge_attr.size(1)), device=token_index.device
    )

    for i in range(order * (order - 1) // 2):
        # NOTE: Example k = 3: (0, 1), (0, 2), (1, 2)
        begin = max(((i + 1) - (order - 1), 0))
        end = min(((i + 1), order - 1))

        index_match: torch.Tensor = (
            token_index[(begin, end), :].T.unsqueeze(-1) == edge_index
        )
        idx = index_match.all(1).nonzero().T
        distributed_edge_features[idx[0]] += edge_attr[idx[1]]

    return distributed_edge_features


class TripletTokenizer(nn.Module):
    def __init__(
        self,
        pe_encoder,
        atom_encoder,
        bond_encoder,
        max_degree,
        eigen_dim,
        degree_dim,
        embed_dim,
        bias,
    ):
        super().__init__()
        self.eigen_embed = pe_encoder
        self.eigen_encoder = ConcatEncoder(3, eigen_dim, embed_dim, bias)
        self.degree_embed = nn.Embedding(max_degree, degree_dim)
        self.degree_encoder = ConcatEncoder(3, degree_dim, embed_dim, bias)
        self.atom_encoder = atom_encoder
        self.node_encoder = ConcatEncoder(3, embed_dim, embed_dim, bias)
        self.bond_encoder = bond_encoder
        self.graph_token = nn.Parameter(torch.zeros((1, 1, embed_dim)))

    def forward(self, data):
        token_index, real_edges = data.token_index, data.real_edges
        x = self.atom_encoder(data.x)
        d = self.degree_embed(data.degrees)
        l = self.eigen_embed(data.eigvals, data.eigvecs, data.edge_index, data.batch)

        e = self.bond_encoder(data.edge_attr)
        e = distribute_edge_features(data.edge_index, e, token_index)

        token_embed = (
            e
            + self.eigen_encoder(l, token_index)
            + self.degree_encoder(d, token_index)
            + self.node_encoder(x, token_index)
        )

        batch = data.batch[token_index[0]]
        token_embed, mask = to_dense_batch(token_embed, batch)

        graph_token = self.graph_token.repeat(token_embed.size(0), 1, 1)
        token_embed = torch.cat([graph_token, token_embed], 1)

        padding_mask = torch.zeros(
            (mask.size(0), mask.size(1) + 1), dtype=mask.dtype, device=mask.device
        )
        padding_mask[:, 0] = 1
        padding_mask[:, 1:] = mask

        return token_embed, padding_mask


class LWLTransformer(ModelBase):
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

        self.tokenizer = PairTokenizer(
            pe_encoder,
            modules["node"],
            modules["edge"],
            embed_dim,
            bias,
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
        self.num_heads = num_heads

    def forward(self, data, return_graph=False, return_node=False, return_edge=False):
        x, mask = self.tokenizer(data)
        attn_mask = ~mask
        x = self.transformer(x, attn_mask)

        if return_graph:
            return x[:, 0]
        elif return_node:
            return x[:, 1:][mask[:, 1:]][~data.real_edges]
        elif return_edge:
            return x[:, 1:][mask[:, 1:]][data.real_edges]

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
