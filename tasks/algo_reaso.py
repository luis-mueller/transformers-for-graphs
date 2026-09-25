import networkx as nx
import random
import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from ogb.lsc import PCQM4Mv2Evaluator
from networkx.algorithms import tree
from tasks.utils import f1, MLP


def generate_graph(num_nodes, num_connect=1, p=0.1):
    graph = nx.erdos_renyi_graph(num_nodes, p=p)
    cc = list(nx.connected_components(graph))
    if len(cc) >= 2:
        for i in range(len(cc)):
            for _ in range(num_connect):
                j = random.choice([j for j in range(len(cc)) if j != i])
                a = random.choice(list(cc[i]))
                b = random.choice(list(cc[j]))
                graph.add_edge(a, b)
    return graph


def mst_graph(num_nodes, num_connect, p):
    while True:
        graph = generate_graph(num_nodes, num_connect, p)
        if len(graph.edges) < 1:
            continue

        weight_dict = {
            e: {"edge_attr": round(random.uniform(0, 10), 6)} for e in graph.edges
        }

        weights = [v["edge_attr"] for k, v in weight_dict.items()]
        if len(weights) == len(set(weights)):
            nx.set_edge_attributes(graph, weight_dict)
            edges = [
                (u, v)
                for u, v, _ in tree.minimum_spanning_edges(graph, weight="edge_attr")
            ]
            return graph, edges


def mst(num_nodes, num_connect=1, p=0.1):
    graph, edges = mst_graph(num_nodes, num_connect, p)
    data = from_networkx(graph)
    data.y = torch.zeros(data.edge_index.size(1), dtype=torch.long)

    for i in range(data.edge_index.size(1)):
        if (data.edge_index[0, i], data.edge_index[1, i]) in edges or (
            data.edge_index[1, i],
            data.edge_index[0, i],
        ) in edges:
            data.y[i] = 1
    data.edge_attr = data.edge_attr[:, None]
    return data


def bridges_graph(num_nodes, num_connect, p):
    graph = generate_graph(num_nodes, num_connect, p)
    edges = list(nx.bridges(graph))
    return graph, edges


def bridges(num_nodes, num_connect=1, p=0.1):
    graph, edges = bridges_graph(num_nodes, num_connect, p)
    data = from_networkx(graph)
    data.y = torch.zeros(data.edge_index.size(1), dtype=torch.long)

    for i in range(data.edge_index.size(1)):
        if (data.edge_index[0, i], data.edge_index[1, i]) in edges or (
            data.edge_index[1, i],
            data.edge_index[0, i],
        ) in edges:
            data.y[i] = 1

    return data


def cycles(num_nodes, num_connect=1, p=0.1):
    graph, _ = bridges_graph(num_nodes, num_connect, p)
    cycle_nodes = list(set([el for cycle in nx.simple_cycles(graph) for el in cycle]))
    cycle_idx = torch.tensor(cycle_nodes, dtype=torch.long)
    data = from_networkx(graph)
    data.y = torch.zeros(num_nodes, dtype=torch.long)
    data.y[cycle_idx] = 1
    return data


def flow_graph(num_nodes, num_connect, p):
    graph = generate_graph(num_nodes, num_connect, p).to_directed()
    weight_dict = {
        e: {"edge_attr": round(random.uniform(0, 3), 2)} for e in graph.edges
    }
    nx.set_edge_attributes(graph, weight_dict)
    nodes = list(range(num_nodes))
    source = random.choice(nodes)
    nodes = [n for n in nodes if n != source]
    sink = random.choice(nodes)
    value = nx.flow.maximum_flow_value(graph, source, sink, capacity="edge_attr")
    return graph, source, sink, value


def flow(num_nodes, num_connect=1, p=0.1):
    graph, source, sink, value = flow_graph(num_nodes, num_connect, p)
    data = from_networkx(graph)
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    data.y = torch.tensor(value)
    data.x[source] = 1
    data.x[sink] = 2
    data.edge_attr = data.edge_attr[:, None]
    return data


ALGORITHMS = {
    "bridges": bridges,
    "cycles": cycles,
    "mst": mst,
    "flow": flow,
}


CONFIG = {"bridges": (1, 0.05), "cycles": (1, 0.05), "mst": (3, 0.1), "flow": (2, 0.05)}


class AlgoReaso(InMemoryDataset):
    def __init__(
        self,
        root,
        name,
        num_nodes,
        num_samples,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        validation=False,
        test=False,
    ):
        if validation:
            root = os.path.join(root, f"{name}_{num_nodes}_val")
        elif test:
            root = os.path.join(root, f"{name}_{num_nodes}_test")
        else:
            root = os.path.join(root, f"{name}_{num_nodes}")

        self.name = name
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        data_list = [
            ALGORITHMS[self.name](self.num_nodes, *CONFIG[self.name])
            for _ in tqdm.tqdm(range(self.num_samples))
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


def load_bridges(
    root,
    embed_dim,
    bias,
    pre_transform,
    transform,
):
    train_dataset = AlgoReaso(
        root, "bridges", 16, 4000000, pre_transform=pre_transform, transform=transform
    )
    valid_dataset = AlgoReaso(
        root,
        "bridges",
        16,
        10000,
        pre_transform=pre_transform,
        transform=transform,
        validation=True,
    )
    test_datasets = {
        f"test_{n}": AlgoReaso(root, "bridges", n, 1000, pre_transform=pre_transform, transform=transform, test=True)
        for n in [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    }
    node_encoder = None
    edge_encoder = None
    decoder = MLP(embed_dim, 2, bias)

    def is_better(best_score, val_metrics):
        better = best_score is None or best_score < val_metrics["bridges_valid_f1"]
        return (
            better,
            val_metrics["bridges_valid_f1"],
            val_metrics["bridges_valid_f1"] == 1.0,
        )

    return (
        {"train": train_dataset, "valid": valid_dataset, **test_datasets},
        {"node": node_encoder, "edge": edge_encoder, "decoder": decoder},
        {
            "loss": lambda x, batch: F.cross_entropy(x, batch.y),
            "metric": ("f1", f1),
            "is_better": is_better,
        },
    )


def load_cycles(
    root,
    embed_dim,
    bias,
    pre_transform,
    transform,
):
    train_dataset = AlgoReaso(
        root, "cycles", 16, 10000, pre_transform=pre_transform, transform=transform
    )
    test_dataset = AlgoReaso(
        root,
        "cycles",
        16,
        5000,
        pre_transform=pre_transform,
        transform=transform,
        validation=True,
    )
 
    node_encoder = None
    edge_encoder = None
    decoder = MLP(embed_dim, 2, bias)

    return (
        {"train": train_dataset, "valid": test_dataset, "test": test_dataset},
        {"node": node_encoder, "edge": edge_encoder, "decoder": decoder},
        {
            "loss": lambda x, batch: F.cross_entropy(x, batch.y),
            "metric": ("f1", f1),
        },
    )


def load_mst(
    root,
    embed_dim,
    bias,
    pre_transform,
    transform,
):
    train_dataset = AlgoReaso(
        root, "mst", 16, 1000000, pre_transform=pre_transform, transform=transform
    )
    valid_dataset = AlgoReaso(
        root,
        "mst",
        16,
        10000,
        pre_transform=pre_transform,
        transform=transform,
        validation=True,
    )
    test_dataset = AlgoReaso(
        root, "mst", 64, 10000, pre_transform=pre_transform, transform=transform
    )
    node_encoder = None
    edge_encoder = nn.Linear(1, embed_dim)
    decoder = MLP(embed_dim, 2, bias)

    def is_better(best_score, val_metrics):
        better = best_score is None or best_score < val_metrics["mst_valid_f1"]
        return better, val_metrics["mst_valid_f1"], val_metrics["mst_valid_f1"] == 1.0

    return (
        {"train": train_dataset, "valid": valid_dataset, "test": test_dataset},
        {"node": node_encoder, "edge": edge_encoder, "decoder": decoder},
        {
            "loss": lambda x, batch: F.cross_entropy(x, batch.y),
            "metric": ("f1", f1),
            "is_better": is_better,
        },
    )


def load_flow(
    root,
    embed_dim,
    bias,
    pre_transform,
    transform,
):
    train_dataset = AlgoReaso(
        root, "flow", 16, 4000000, pre_transform=pre_transform, transform=transform
    )
    valid_dataset = AlgoReaso(
        root,
        "flow",
        16,
        10000,
        pre_transform=pre_transform,
        transform=transform,
        validation=True,
    )
    test_datasets = {
        f"test_{n}": AlgoReaso(root, "flow", n, 1000, pre_transform=pre_transform, transform=transform, test=True)
        for n in [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    }

    node_encoder = nn.Embedding(3, embed_dim)
    edge_encoder = nn.Linear(1, embed_dim)
    decoder = MLP(
        embed_dim, 1, bias, squeeze=True, project_down=True, activation="gelu"
    )

    evaluator = PCQM4Mv2Evaluator()
    metric = (
        "mae",
        lambda y_pred, y_true: evaluator.eval(
            {"y_pred": torch.cat(y_pred), "y_true": torch.cat(y_true)}
        )["mae"],
    )

    def is_better(best_score, val_metrics):
        better = best_score is None or best_score > val_metrics["flow_valid_mae"]
        return (
            better,
            val_metrics["flow_valid_mae"],
            val_metrics["flow_valid_mae"] == 0.0,
        )

    return (
        {"train": train_dataset, "valid": valid_dataset, **test_datasets},
        {"node": node_encoder, "edge": edge_encoder, "decoder": decoder},
        {
            "loss": lambda x, batch: F.l1_loss(x, batch.y),
            "metric": metric,
            "is_better": is_better,
        },
    )
