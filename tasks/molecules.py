import torch
import torch.nn.functional as F
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from tasks.utils import MLP
import numpy


def load_pcqm4mv2(
    root,
    embed_dim,
    bias,
    pre_transform,
    transform,
):  # 4M graphs
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
    dataset = PygPCQM4Mv2Dataset(root, pre_transform=pre_transform, transform=transform)
    splits = dataset.get_idx_split()

    train_data = dataset[splits["train"]].shuffle()
    train_dataset = train_data[10000:]
    valid_dataset = train_data[:10000]
    test_dataset = dataset[splits["valid"]]

    node_encoder = AtomEncoder(embed_dim)
    edge_encoder = BondEncoder(embed_dim)

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
        better = best_score is None or best_score > val_metrics["pcq_valid_mae"]
        return better, val_metrics["pcq_valid_mae"], val_metrics["pcq_valid_mae"] == 0.0

    return (
        {"train": train_dataset, "valid": valid_dataset, "test": test_dataset},
        {"node": node_encoder, "edge": edge_encoder, "decoder": decoder},
        {
            "loss": lambda x, batch: F.l1_loss(x, batch.y),
            "metric": metric,
            "is_better": is_better,
        },
    )
