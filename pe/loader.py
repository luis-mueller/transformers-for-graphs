import torch.nn as nn
from functools import partial
from torch_geometric.transforms import Compose
from pe.random_walk import random_walk_transform, RWSE, RRWP
from pe.laplacian import laplacian_transform, LPE, SPE
from pe.edge import edge_transform
from pe.higher_order import token_index_transform, LWLTransform

PE = ["nope", "rwse", "rrwp", "lpe", "spe"]


def add_transform_args(parser):
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        default=[],
        choices=PE + ["edge", "token_index", "lwl"],
    )

    # Random-Walk
    parser.add_argument("--max_rw_steps", type=int, default=32)

    # Laplacian
    parser.add_argument("--max_eigvals", type=int, default=32)
    parser.add_argument("--large_graph", type=bool, default=False)
    parser.add_argument("--normalized", type=bool, default=True)

    # Edge Transform
    parser.add_argument("--undirected", type=bool, default=False)
    parser.add_argument("--preserve_graph", type=bool, default=True)


def add_pe_args(parser):
    parser.add_argument("--pe", type=str, default="nope", choices=PE)
    parser.add_argument("--rwse_steps", type=int, default=16)
    parser.add_argument("--rrwp_steps", type=int, default=16)
    parser.add_argument("--spe_num_eigvals", type=int, default=8)
    parser.add_argument("--spe_hidden_dim", type=int, default=384)
    parser.add_argument("--spe_inner_dim", type=int, default=384)
    parser.add_argument("--spe_phi_dim", type=int, default=384)
    parser.add_argument("--spe_num_layers_phi", type=int, default=2)
    parser.add_argument("--spe_num_layers_rho", type=int, default=2)
    parser.add_argument("--spe_lower_rank", type=bool, default=True)
    parser.add_argument("--lpe_num_eigvals", type=int, default=32)
    parser.add_argument("--lpe_position_aware", type=bool, default=True)


def load_transform(
    transforms,
    max_rw_steps,
    max_eigvals,
    large_graph,
    normalized,
    undirected,
    preserve_graph,
    task,
):
    transform_list = []

    if "edge" in transforms:
        transform_list.append(
            partial(
                edge_transform,
                preserve_nodes=True,
                undirected=undirected,
                preserve_graph=preserve_graph,
            )
        )

    if "token_index" in transforms:
        transform_list.append(token_index_transform)

    if "rwse" in transforms:
        transform_list.append(
            partial(
                random_walk_transform,
                max_rw_steps=max_rw_steps,
                diagonal=True,
            )
        )
    if "rrwp" in transforms:
        transform_list.append(
            partial(
                random_walk_transform,
                max_rw_steps=max_rw_steps,
                diagonal=False,
                task=task,
            )
        )
    if "lpe" in transforms or "spe" in transforms:
        transform_list.append(
            partial(
                laplacian_transform,
                normalized=normalized,
                large_graph=large_graph,
                max_eigvals=max_eigvals,
            )
        )

    if "lwl" in transforms:
        transform_list.append(LWLTransform(2))

    if len(transform_list) == 0:
        return None

    return Compose(transform_list)


def load_transform_from_args(args, task=None):
    return load_transform(
        args.transforms,
        args.max_rw_steps,
        args.max_eigvals,
        args.large_graph,
        args.normalized,
        args.undirected,
        args.preserve_graph,
        task,
    )


def load_pe(
    pe,
    rwse_steps,
    rrwp_steps,
    lpe_num_eigvals,
    lpe_position_aware,
    spe_num_layers_phi,
    spe_num_eigvals,
    spe_phi_dim,
    spe_inner_dim,
    spe_hidden_dim,
    spe_num_layers_rho,
    spe_lower_rank,
    embed_dim,
    num_heads,
):
    if pe == "nope":

        class NoPE(nn.Module):
            def forward(*args, **kwargs):
                return None, None

        return NoPE()

    elif pe == "rwse":
        return RWSE(
            rwse_steps,
            embed_dim,
        )

    elif pe == "rrwp":
        return RRWP(
            rrwp_steps,
            num_heads,
            embed_dim,
        )

    elif pe == "lpe":
        return LPE(
            embed_dim,
            lpe_num_eigvals,
            lpe_position_aware,
        )

    elif pe == "spe":
        return SPE(
            embed_dim,
            spe_num_layers_phi,
            spe_num_eigvals,
            spe_phi_dim,
            spe_inner_dim,
            spe_hidden_dim,
            spe_num_layers_rho,
            spe_lower_rank,
        )


def load_pe_from_args(args, embed_dim, num_heads):
    return load_pe(
        args.pe,
        args.rwse_steps,
        args.rrwp_steps,
        args.lpe_num_eigvals,
        args.lpe_position_aware,
        args.spe_num_layers_phi,
        args.spe_num_eigvals,
        args.spe_phi_dim,
        args.spe_inner_dim,
        args.spe_hidden_dim,
        args.spe_num_layers_rho,
        args.spe_lower_rank,
        embed_dim,
        num_heads,
    )
