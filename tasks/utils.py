import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch_geometric.utils import to_dense_batch


def mae(preds: torch.Tensor, targets: torch.Tensor):
    return (preds - targets).abs().mean(dtype=float).item()


def accuracy(preds: torch.Tensor, targets: torch.Tensor):
    return (preds.softmax(-1).argmax(-1) == targets).mean(dtype=float).item()


def f1(preds: list[torch.Tensor], targets: list[torch.Tensor]):
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    if preds.dtype != torch.long:
        preds = preds.softmax(-1).argmax(-1)
    return float(
        f1_score(
            targets.to(dtype=torch.float32).detach().cpu().numpy(),
            preds.to(dtype=torch.float32).detach().cpu().numpy(),
            average="macro",
            zero_division=0,
        )
    )


def ap(preds: torch.Tensor, targets: torch.Tensor):
    preds = torch.sigmoid(preds)
    return sum(
        [
            float(
                average_precision_score(
                    targets[:, i].to(dtype=torch.float32).detach().cpu().numpy(),
                    preds[:, i].to(dtype=torch.float32).detach().cpu().numpy(),
                )
            )
            for i in range(preds.size(1))
        ]
    ) / preds.size(1)


def rocauc(preds: torch.Tensor, targets: torch.Tensor):
    preds = preds.softmax(-1).argmax(-1)
    return float(
        roc_auc_score(
            targets.to(dtype=torch.float32).detach().cpu().numpy(),
            preds.to(dtype=torch.float32).detach().cpu().numpy(),
        )
    )


def first_pool(data):
    x, _ = to_dense_batch(data.x, data.batch)
    return x[torch.arange(x.size(0), device=x.device), data.mapping]


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim,
        output_dim,
        bias,
        squeeze=False,
        project_down=False,
        activation="gelu",
    ):
        super().__init__()
        self.squeeze = squeeze
        activation = nn.GELU if activation == "gelu" else nn.ReLU

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(
                embed_dim, embed_dim // 2 if project_down else embed_dim, bias=bias
            ),
            activation(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(
                embed_dim // 2 if project_down else embed_dim,
                embed_dim // 4 if project_down else embed_dim,
                bias=bias,
            ),
            activation(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(
                embed_dim // 4 if project_down else embed_dim, output_dim, bias=bias
            ),
        )

    def forward(self, x):
        x = self.nn(x)
        if self.squeeze:
            x = x.squeeze()
        return x
