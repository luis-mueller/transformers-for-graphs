import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_dense_batch, get_laplacian, to_scipy_sparse_matrix
from loguru import logger


def laplacian_transform(data, normalized, large_graph, max_eigvals):
    if data.num_nodes > 1000:
        logger.info("Skipping laplacian transform for graph with more than 1K nodes")
        data.eigvals = torch.empty((data.num_nodes, max_eigvals))
        data.eigvecs = torch.empty((data.num_nodes, max_eigvals))
        return data
    if hasattr(data, "eigvals") and hasattr(data, "eigvecs"):
        return data
    A = torch.zeros((data.num_nodes, data.num_nodes))
    A[data.edge_index[0], data.edge_index[1]] = 1

    if normalized:
        D12 = torch.diag(A.sum(1).clip(1) ** -0.5)
        I = torch.eye(A.size(0))
        L = I - D12 @ A @ D12
    else:
        D = torch.diag(A.sum(1))
        L = D - A

    if large_graph:
        L_edge_index, L_edge_weight = get_laplacian(
            data.edge_index, num_nodes=data.num_nodes, normalization="sym"
        )
        L = to_scipy_sparse_matrix(L_edge_index, L_edge_weight)

        eigvals, eigvecs = np.linalg.eigh(L.todense())
        eigvals = torch.from_numpy(np.real(eigvals))
        eigvecs = torch.from_numpy(eigvecs).float()
    else:
        eigvals, eigvecs = torch.linalg.eigh(L)

    eigvals_size = eigvals.size(0)
    if large_graph:
        idx1 = torch.argsort(eigvals)[: max_eigvals // 2]
        idx2 = torch.argsort(eigvals, descending=True)[: max_eigvals // 2]
        idx = torch.cat([idx1, idx2])
    else:
        idx = torch.argsort(eigvals)[:max_eigvals]

    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals, eigvecs = eigvals[:eigvals_size], eigvecs[:, :eigvals_size]

    eigvals = torch.real(eigvals).clamp_min(0)
    eigvals = eigvals.unsqueeze(0).repeat(data.num_nodes, 1)
    if data.num_nodes < max_eigvals:
        eigvals = F.pad(eigvals, (0, max_eigvals - data.num_nodes), value=float("nan"))
        eigvecs = F.pad(eigvecs, (0, max_eigvals - data.num_nodes), value=float("nan"))

    data.eigvals = eigvals
    data.eigvecs = eigvecs

    return data


def nan_to_zero(mat):
    empty_mask = torch.isnan(mat)
    mat[empty_mask] = 0
    return mat


class EigenMLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.bn(x.reshape(-1, x.size(-1))).reshape(x.size())
        return self.dropout(F.relu(x))


class EigenMLP(nn.Module):
    def __init__(self, num_layers, embed_dim, out_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EigenMLPLayer(1 if i == 0 else embed_dim, embed_dim, dropout)
                for i in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(embed_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out_proj(x)
        return self.dropout(x)


class EigenGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                torch_geometric.nn.conv.GINConv(
                    nn.Sequential(
                        nn.Linear(input_dim if _ == 0 else hidden_dim, hidden_dim),
                        nn.ReLU(),
                    ),
                    node_dim=0,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            z = layer(x, edge_index)
            if i > 0:
                x = z + x
            else:
                x = z
        return self.out_proj(x)


class SPE(nn.Module):
    def __init__(
        self,
        embed_dim,
        spe_num_layers_phi,
        spe_num_eigvals,
        spe_phi_dim,
        spe_inner_dim,
        spe_hidden_dim,
        spe_num_layers_rho,
        lower_rank,
    ):
        super().__init__()
        self.psi = EigenMLP(spe_num_layers_phi, spe_phi_dim, spe_inner_dim, dropout=0)
        self.gin = EigenGIN(
            input_dim=spe_inner_dim,
            hidden_dim=spe_hidden_dim,
            output_dim=embed_dim,
            num_layers=spe_num_layers_rho,
        )
        self.lower_rank = lower_rank
        self.num_eigvals = spe_num_eigvals

    def forward(self, data, is_training):
        eigvals = nan_to_zero(data.eigvals[:, : self.num_eigvals])
        eigvecs = nan_to_zero(data.eigvecs[:, : self.num_eigvals])

        eigvals, _ = to_dense_batch(eigvals, data.batch)  # B, N, M
        eigvals = eigvals[:, 0, :].unsqueeze(-1)  # B, M, 1

        eigvals = self.psi(eigvals)  # B, M, N_psi

        eigvecs, mask = to_dense_batch(eigvecs, data.batch)  # B, N, M

        if self.lower_rank is not None:
            outer_eigvecs = eigvecs[:, : self.lower_rank]
        else:
            outer_eigvecs = eigvecs

        eigen_tensor = torch.einsum(
            "bijk,blj->bilk",
            eigvecs.unsqueeze(-1) * eigvals.unsqueeze(1),  # B, N, M, N_psi
            outer_eigvecs,
        )  # B, N, LN, N_psi
        pe = eigen_tensor[mask]  # N_sum, LN, N_psi
        pe = self.gin(pe, data.edge_index).sum(1)
        return pe, None


class LPE(nn.Module):
    def __init__(
        self,
        embed_dim,
        lpe_num_eigvals,
        position_aware,
    ):
        super().__init__()
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.phi = nn.Sequential(
            nn.Linear(2, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        if position_aware:
            self.eps = nn.Parameter(1e-12 * torch.arange(lpe_num_eigvals)[None])
        self.position_aware = position_aware
        self.lpe_num_eigvals = lpe_num_eigvals

    def forward(self, data, is_training):
        eigvecs = data.eigvecs[:, : self.lpe_num_eigvals]
        eigvals = data.eigvals[:, : self.lpe_num_eigvals]

        if is_training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)

        if self.position_aware:
            eigvals = eigvals + self.eps[:, : self.lpe_num_eigvals]

        x = torch.stack((eigvecs, eigvals), 2)
        empty_mask = torch.isnan(x)
        x[empty_mask] = 0

        eigen_embed = self.phi(x)
        lpe_pe = self.rho(eigen_embed.sum(1))
        return lpe_pe, None
