import torch
import torch.nn as nn

STDDEV = 0.02
CTX_SIZES = [8, 16, 32, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192, 16384]


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=STDDEV)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=STDDEV)


def get_context_size(batch):
    max_num_nodes = batch.unique(return_counts=True)[1].max().item()
    return (max_num_nodes + 3) // 4 * 4


def add_to_attn_mask(attn_mask, relative_pe, sparse_pos, batch):
    batch_size = attn_mask.size(0)
    context_size = attn_mask.size(1)
    repeats = batch.unique(return_counts=True)[1] ** 2
    offsets = (context_size**2 * torch.arange(batch_size)).repeat_interleave(repeats)
    idx = sparse_pos + offsets
    attn_mask_view = attn_mask.view(-1, attn_mask.size(-1))
    attn_mask_view[idx] = attn_mask_view[idx] + relative_pe
