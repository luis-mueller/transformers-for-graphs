import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import LRGBDataset
from tasks.utils import f1, MLP


def center_features(dataset, x_mu=None, x_sigma=None, e_mu=None, e_sigma=None):
    if x_mu is None:
        x_mu, x_sigma = dataset.x.mean(0, keepdim=True), dataset.x.std(0, keepdim=True)
        e_mu, e_sigma = dataset.edge_attr.mean(0, keepdim=True), dataset.edge_attr.std(
            0, keepdim=True
        )
    dataset.x = (dataset.x - x_mu) / x_sigma
    dataset.edge_attr = (dataset.edge_attr - e_mu) / e_sigma
    return dataset, x_mu, x_sigma, e_mu, e_sigma


def load_pascal(root, embed_dim, bias, pre_transform, transform):  # 10K graphs
    train_dataset, x_mu, x_sigma, e_mu, e_sigma = center_features(
        LRGBDataset(
            root,
            "PascalVOC-SP",
            split="train",
            pre_transform=pre_transform,
            transform=transform,
        )
    )
    valid_dataset, _, _, _, _ = center_features(
        LRGBDataset(
            root,
            "PascalVOC-SP",
            split="val",
            pre_transform=pre_transform,
            transform=transform,
        ),
        x_mu=x_mu,
        x_sigma=x_sigma,
        e_mu=e_mu,
        e_sigma=e_sigma,
    )
    test_dataset, _, _, _, _ = center_features(
        LRGBDataset(
            root,
            "PascalVOC-SP",
            split="test",
            pre_transform=pre_transform,
            transform=transform,
        ),
        x_mu=x_mu,
        x_sigma=x_sigma,
        e_mu=e_mu,
        e_sigma=e_sigma,
    )

    node_encoder = nn.Linear(train_dataset.num_node_features, embed_dim)
    edge_encoder = nn.Linear(train_dataset.num_edge_features, embed_dim)
    decoder = MLP(embed_dim, train_dataset.num_classes, bias)

    def is_better(best_score, val_metrics):
        better = best_score is None or best_score < val_metrics["pascal_valid_f1"]
        return (
            better,
            val_metrics["pascal_valid_f1"],
            val_metrics["pascal_valid_f1"] == 1.0,
        )

    return (
        {"train": train_dataset, "valid": valid_dataset, "test": test_dataset},
        {"node": node_encoder, "edge": edge_encoder, "decoder": decoder},
        {
            "loss": lambda x, batch: F.cross_entropy(x, batch.y),
            "metric": ("f1", f1),
            "is_better": is_better,
        },
    )


def load_coco(root, embed_dim, bias, pre_transform, transform):  # 120K graphs
    train_dataset, x_mu, x_sigma, e_mu, e_sigma = center_features(
        LRGBDataset(
            root,
            "COCO-SP",
            split="train",
            pre_transform=pre_transform,
            transform=transform,
        )
    )
    valid_dataset, _, _, _, _ = center_features(
        LRGBDataset(
            root,
            "COCO-SP",
            split="val",
            pre_transform=pre_transform,
            transform=transform,
        ),
        x_mu=x_mu,
        x_sigma=x_sigma,
        e_mu=e_mu,
        e_sigma=e_sigma,
    )
    test_dataset, _, _, _, _ = center_features(
        LRGBDataset(
            root,
            "COCO-SP",
            split="test",
            pre_transform=pre_transform,
            transform=transform,
        ),
        x_mu=x_mu,
        x_sigma=x_sigma,
        e_mu=e_mu,
        e_sigma=e_sigma,
    )
    node_encoder = nn.Linear(train_dataset.num_node_features, embed_dim)
    edge_encoder = nn.Linear(train_dataset.num_edge_features, embed_dim)
    decoder = MLP(embed_dim, train_dataset.num_classes, bias)

    def is_better(best_score, val_metrics):
        better = best_score is None or best_score < val_metrics["coco_valid_f1"]
        return better, val_metrics["coco_valid_f1"], val_metrics["coco_valid_f1"] == 1.0

    return (
        {"train": train_dataset, "valid": valid_dataset, "test": test_dataset},
        {"node": node_encoder, "edge": edge_encoder, "decoder": decoder},
        {
            "loss": lambda x, batch: F.cross_entropy(x, batch.y),
            "metric": ("f1", f1),
            "is_better": is_better,
        },
    )
