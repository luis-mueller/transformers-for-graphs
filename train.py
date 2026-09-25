import os
import pandas as pd
import torch
import argparse
import torch
import copy
import torch.nn as nn
from loguru import logger
from model import (
    load_model_size_from_args,
    load_model_from_args,
    add_model_args,
    ModelWrapper,
)
from torch_geometric.seed import seed_everything
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import (
    metric_dict_to_str,
    log_metrics,
    ResourceProfile,
    accelerator_setup,
    setup_run,
    ddp_setup,
    configure_optimizers,
)

from pe import (
    load_transform_from_args,
    load_pe_from_args,
    add_pe_args,
    add_transform_args,
)
from tasks import load_task_from_args, add_task_args, override_runtime_behavior
from evaluation import load_evaluation_from_args, add_evaluation_args


def load_everything(args, device_count=1, main_process=True):
    transform = load_transform_from_args(args)
    _, embed_dim, _, num_heads = load_model_size_from_args(args)
    pe_encoder = load_pe_from_args(args, embed_dim, num_heads)

    data, modules, funcs, level = load_task_from_args(
        args,
        embed_dim,
        args.bias,
        transform,
        device_count,
        main_process,
    )

    backbone = load_model_from_args(args, modules, pe_encoder)
    orig_model = ModelWrapper(backbone, modules["decoder"], level)
    return data, funcs, orig_model


def test_all_splits(evaluate, task, data, funcs, model, ctx, device):
    test_metrics = {}
    for test_split in [k for k in data.keys() if k.startswith("test")]:
        _test_metrics = evaluate(
            task, data, funcs, model, test_split, ctx, device
        )
        test_metrics.update(_test_metrics)
    return test_metrics


def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sets the random seed used across the complete training.",
    )
    parser.add_argument(
        "--disable_runtime_override",
        action="store_true",
        help="Disables task-specific runtime overrides (e.g., number of steps, or batch size). Set to run with custom runtime args.",
    )

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100000,
        help="Setting the number of steps a model runs for.",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=20000,
        help="The number of warm up iterations for the learning rate scheduler.",
    )
    parser.add_argument(
        "--gradient_norm",
        type=float,
        default=5.0,
        help="The gradient norm for gradient updates in the backward pass.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="")

    # IO
    parser.add_argument(
        "--log_every",
        type=int,
        default=1000,
        help="determines the interval in steps for which results should be logged to a .txt file.",
    )
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument(
        "--test_during_training",
        action="store_true",
        help="Sets the option to provide results on the test set during training. If not set, the test result will only be obtained at the end of training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="gives an absolute or relative path to a checkpoint file to be used for this run.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile the model. We use a generic fullgraph=False.",
    )
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="Whether to call DDP with find_unused_parameters=True",
    )
    add_task_args(parser)
    add_model_args(parser)
    add_pe_args(parser)
    add_transform_args(parser)
    add_evaluation_args(parser)
    args = parser.parse_args()

    logger.info(vars(args))

    if not args.disable_runtime_override:
        args = override_runtime_behavior(args)

    seed_everything(args.seed)

    ddp_setup()
    ctx, _, device = setup_run()
    device, device_id, device_count, main_process = accelerator_setup()

    data, funcs, orig_model = load_everything(args, device_count, main_process)
    logger.info(orig_model)
    orig_model.reset_parameters()
    orig_model = orig_model.to(device)
    param_count = sum(p.numel() for p in orig_model.parameters())
    logger.info(f"Model params: {param_count}")

    if device_count > 1:
        logger.info("Creating DDP module")
        model = DDP(
            orig_model,
            device_ids=[device_id],
            find_unused_parameters=args.find_unused_parameters,
        )
    else:
        model = orig_model

    if args.compile:
        orig_model.backbone.transformer = torch.compile(orig_model.backbone.transformer)

    optimizer = configure_optimizers(
        model,
        args.weight_decay,
        args.learning_rate,
        (0.9, 0.95),
        device,
    )

    def get_loss():
        batch = data["train"].sample().to(device)

        with ctx:
            preds = model(batch)
        return funcs["loss"](preds, batch)

    evaluate = load_evaluation_from_args(args)

    resource_profile = ResourceProfile()
    log_every = 4096

    logger.info(f"Starting training for {args.num_steps} steps 🍿")
    warmup_iters = args.warmup_iters

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, step / warmup_iters)
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps - warmup_iters
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        [warmup_iters],
    )
    logger.info(f"Cosine schedule with {warmup_iters} warm-up iters.")

    best_loss = None
    test_metrics = {}
    state_dict = {}
    best_model = None

    loss_window = []

    resource_profile.activate()

    for step in range(args.num_steps):
        train_loss = get_loss()
        train_loss.backward()
        _loss = train_loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if main_process:
            loss_window.append(_loss)

        if (
            main_process
            and (step + 1) % log_every == 0
        ):
            logger.info("Running validation")
            model.eval()
            val_metrics = evaluate(args.task, data, funcs, model, "valid", ctx, device)
            better, score, best_possible = funcs["is_better"](best_loss, val_metrics)

            if better:
                best_loss = score
                if args.checkpoint is not None:
                    torch.save(model.state_dict(), args.checkpoint)

                if args.test_during_training:
                    test_metrics = test_all_splits(evaluate, args.task, data, funcs, model, ctx, device)
                else:
                    best_loss = score
                    best_model = copy.deepcopy(model)

            lr = optimizer.param_groups[0]["lr"]
            loss = sum(loss_window) / len(loss_window)
            log_metrics(step, lr, loss, best_loss, val_metrics, test_metrics)
            loss_window = []
            model.train()

            if best_possible:
                logger.info("Best possible validation score reached: Stopping...")
                break
    
    timing, memory = resource_profile.stop(step + 1)
    logger.info("Training complete ✨")

    if main_process:
        if not args.test_during_training:
            model = best_model
            test_metrics = test_all_splits(evaluate, args.task, data, funcs, model, ctx, device)

        logger.info(f"Final results: {metric_dict_to_str(test_metrics)}")

        if args.path is not None:
            results = [
                {
                    "best_val_score": best_loss,
                    **val_metrics,
                    **test_metrics,
                    "timing": timing,
                    "memory": memory,
                    **vars(args),
                }
            ]
            results[0]["achieved_perfect_score_at"] = step

            logger.info(f"Logging results to {args.path}")
            if os.path.exists(path := args.path):
                pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
            else:
                pd.DataFrame(results).to_csv(path, header=True, index=False)

    if device_count > 1:
        destroy_process_group()


if __name__ == "__main__":
    main()
