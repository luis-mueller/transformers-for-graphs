import torch
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj

from model.utils import get_context_size
from loguru import logger


def normalized_adjacency(data, device):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    edge_index, _ = add_remaining_self_loops(edge_index)
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[edge_index[0], edge_index[1]] = 1

    deg_inv_sum = torch.diag_embed(adj.sum(dim=1))
    deg_inv = torch.nan_to_num(1 / deg_inv_sum, posinf=0)
    return deg_inv @ adj


def compute_sparse_pos(num_nodes):
    return torch.arange(num_nodes).repeat_interleave(
        num_nodes
    ) * num_nodes + torch.arange(num_nodes).repeat(num_nodes)


def random_walk_transform(data, max_rw_steps, diagonal=True, device="cpu"):
    if data.num_nodes > 1000:
        logger.info("Skipping random-walk transform for graph with more than 1K nodes")
        if diagonal:
            data.rwse = torch.empty((data.num_nodes, max_rw_steps))
        else:
            data.rrwp = torch.empty((data.num_nodes**2, max_rw_steps))
        return data
    P = normalized_adjacency(data, device)
    P_0 = P.clone()

    probs_shape = [P.size(0)]
    if not diagonal:
        probs_shape += [P.size(0)]
    probs_shape += [max_rw_steps]
    probs = torch.empty(probs_shape).to(device)

    for k in range(max_rw_steps):
        if diagonal:
            probs[:, k] = torch.diagonal(P, 0, -2, -1)
        else:
            probs[:, :, k] = P
        P = P @ P_0

    if diagonal:
        data.rwse = probs
    else:
        data.rrwp = probs.flatten(0, 1)
        data.sparse_pos = compute_sparse_pos(probs.size(0))
    return data


class RWSE(nn.Module):
    def __init__(self, rwse_steps, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.rwse_steps = rwse_steps
        self.encoder = nn.Sequential(
            nn.Linear(rwse_steps, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, data, is_training):
        return self.encoder(data.rwse[:, : self.rwse_steps]), None


class RRWP(torch.nn.Module):
    def __init__(self, rrwp_steps, num_heads, embed_dim):
        super().__init__()
        self.edge_enc = nn.Linear(embed_dim, num_heads)
        self.rrwp_steps = rrwp_steps
        self.encoder = nn.Sequential(
            nn.Linear(rrwp_steps, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, data, is_training):
        return None, self.edge_enc(self.encoder(data.rrwp[:, : self.rrwp_steps]))
