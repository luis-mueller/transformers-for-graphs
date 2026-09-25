import torch
from torch_geometric.utils import add_self_loops


def token_index_transform(data):
    token_index = torch.arange(data.num_nodes).unsqueeze(0)
    token_index = torch.cat(
        [
            token_index.repeat_interleave(data.num_nodes, 1),
            torch.arange(data.num_nodes).repeat(data.num_nodes).unsqueeze(0),
        ],
        dim=0,
    )
    data.token_index = token_index
    return data


def compute_degrees(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    adj[edge_index[0], edge_index[1]] = 1
    return adj.sum(-1)


def compute_lwl_token_index(edge_index, num_nodes, order):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    assert edge_index.size(1) > 0
    if order == 2:
        return edge_index
    elif order == 3:
        nodes = torch.arange(num_nodes)
        edges: torch.Tensor = edge_index.T
        repeated_edges = edges.repeat_interleave(num_nodes, 0)  # E * N x 2
        repeated_nodes = nodes.repeat(len(edges))  # E * N x 1

        candidate_tuples = torch.cat(
            [repeated_edges, repeated_nodes.unsqueeze(-1)], dim=-1
        )  # E * N x 3
        candidate_masks = []

        for begin, end in [(0, 2), (1, 2)]:
            mask = (
                (candidate_tuples[:, (begin, end)].unsqueeze(1) == edges)
                .all(-1)
                .nonzero()[:, 0]
            )
            candidate_masks.append(mask)

        idx = torch.cat(candidate_masks).unique()

        token_index = torch.cat(
            [
                candidate_tuples[idx][:, (0, 1, 2)],
                candidate_tuples[idx][:, (0, 2, 1)],
                candidate_tuples[idx][:, (2, 0, 1)],
            ]
        ).T

        token_index = torch.cat(
            [
                token_index,
                torch.cat([edge_index, edge_index[1].unsqueeze(0)]),
                torch.cat([edge_index[0].unsqueeze(0), edge_index]),
            ],
            -1,
        )

        return token_index
    else:
        raise ValueError(f"Order > 3 is not yet supported.")


def compute_real_edges(edge_index, num_nodes):
    _, attr = add_self_loops(
        edge_index, torch.ones(edge_index.size(1)), fill_value=0, num_nodes=num_nodes
    )
    return attr.to(bool)


class LWLTransform:
    def __init__(self, order):
        self.order = order

    def __call__(self, data):
        data.degrees = compute_degrees(data.edge_index, data.num_nodes)
        data.token_index = compute_lwl_token_index(
            data.edge_index, data.num_nodes, self.order
        )
        data.real_edges = compute_real_edges(data.edge_index, data.num_nodes)
        return data
