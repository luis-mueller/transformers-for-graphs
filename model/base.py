import torch
import torch.nn as nn
from torch_geometric.data import Batch
from abc import ABC, abstractmethod


class ModelBase(nn.Module, ABC):
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
        super().__init__()

    @abstractmethod
    def forward(
        self, data: Batch, return_graph: bool, return_node: bool, return_edge: bool
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def reset_parameters(self):
        pass


class AttentionBase(nn.Module, ABC):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float, bias: bool, **kwargs
    ):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        pass


class ModelWrapper(nn.Module):
    def __init__(self, backbone: ModelBase, decoder: nn.Module, level: int):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.level = level

    def forward(self, data, get_embeddings=False):
        kwargs = {f"return_{self.level}": True}
        embeds = self.backbone(data, **kwargs)
        if get_embeddings:
            return embeds
        return self.decoder(embeds)

    def reset_parameters(self):
        self.backbone.reset_parameters()
