import torch
from torch_geometric.data import Data


def remap_features(num_nodes, edge_index, tuple_index, features):
    outer_shape = features.shape[1:]
    adj = torch.zeros((num_nodes, num_nodes, *outer_shape), dtype=features.dtype)
    adj[edge_index[0], edge_index[1]] = features
    return adj[tuple_index[:, 0], tuple_index[:, 1]]


def edge_transform(
    data,
    preserve_nodes=True,
    preserve_graph=False,
    undirected=True,
):
    if undirected:
        adj = torch.zeros((data.num_nodes, data.num_nodes))
        adj[data.edge_index[0], data.edge_index[1]] = 1
        tuple_index = torch.triu(adj).nonzero()

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            orig_edge_attr = remap_features(
                data.num_nodes, data.edge_index, tuple_index, data.edge_attr
            )
        else:
            orig_edge_attr = None

        if len(data.y) == data.edge_index.size(1):
            y = remap_features(data.num_nodes, data.edge_index, tuple_index, data.y)
        else:
            y = data.y
    else:
        tuple_index = data.edge_index.T
        orig_edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        y = data.y

    orig_edge_index = tuple_index.clone().T

    mask = [True] * tuple_index.size(0)

    if preserve_nodes:
        tuple_index = torch.cat(
            [torch.arange(data.num_nodes)[:, None].repeat(1, 2), tuple_index], 0
        )

        mask = [False] * data.num_nodes + mask

    edge_mask = tuple_index[:, None] == tuple_index[None]
    adj = torch.zeros(tuple_index.size(0), tuple_index.size(0), dtype=torch.long)

    self_loop_mask = torch.logical_and(edge_mask[:, :, 0], edge_mask[:, :, 1])
    edge_index1 = (~self_loop_mask & edge_mask[:, :, 0]).nonzero().T
    edge_index2 = (~self_loop_mask & edge_mask[:, :, 1]).nonzero().T

    adj[edge_index1[0], edge_index1[1]] = 1
    adj[edge_index2[0], edge_index2[1]] = 1 if undirected else 2

    if preserve_graph:
        assert preserve_nodes, "preserve_graph requires preserve_nodes to be True"
        adj[data.edge_index[0], data.edge_index[1]] = adj.max() + 1

    edge_index = adj.nonzero().T
    edge_attr = adj[edge_index[0], edge_index[1]] - 1
    x = torch.tensor(mask).to(torch.long)

    data_dict = dict(
        token_mask=torch.tensor(mask),
        x=x,
        edge_index=edge_index,
        orig_edge_index=orig_edge_index,
        num_nodes=tuple_index.size(0),
        num_edges=edge_index.size(1),
        y=y,
    )

    if (edge_attr > 0).any():
        data_dict["edge_attr"] = edge_attr

    if hasattr(data, "x") and data.x is not None:
        data_dict["x_node"] = data.x
    else:
        data_dict["x_node"] = torch.zeros(
            (data.num_nodes,), device=data.edge_index.device, dtype=torch.long
        )

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data_dict["x_edge"] = orig_edge_attr
    else:
        data_dict["x_edge"] = torch.zeros(
            (data.edge_index.size(1),), device=data.edge_index.device, dtype=torch.long
        )

    return Data(**data_dict)
