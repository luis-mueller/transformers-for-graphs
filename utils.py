import math
import os
import time
import inspect
import torch
import pandas as pd
from loguru import logger
from contextlib import nullcontext
from torch.distributed import init_process_group


def metric_dict_to_str(metric_dict):
    if len(metric_dict) == 0:
        return ""

    return " | ".join([f"{name}: {score:.4f}" for name, score in metric_dict.items()])


def log_metrics(step, lr, loss, best_loss, val_metrics, test_metrics):
    val_metrics = metric_dict_to_str(val_metrics)
    test_metrics = metric_dict_to_str(test_metrics)

    logger.info(
        f"Step: {step} | LR: {lr} | Train loss: {loss:.4f} | Best validation score: {best_loss:.4f} | {val_metrics} | {test_metrics}"
    )


class ResourceProfile:
    def __init__(self):
        logger.info("Setting up resource tracking...")

    def activate(self):
        logger.info("Resource tracking started")
        torch.cuda.synchronize()
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats(device=None)

    def stop(self, total_steps):
        torch.cuda.synchronize()
        total_time = time.time() - self.start_time
        steps_per_s = total_steps / total_time
        memory = torch.cuda.max_memory_allocated(device=None) / (1024**2)

        logger.info(f"Total time: {total_time} s")
        logger.info(f"Avg. {steps_per_s} steps/s ")
        logger.info(f"GPU memory allocation: {memory} MB")
        return total_time, memory


def setup_run():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator 🚀: {device}")

    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float32"
    )
    logger.info(f"Data type: {dtype}")

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    return ctx, ptdtype, device


def ddp_setup():
    if torch.cuda.device_count() > 1:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def accelerator_setup():
    if torch.cuda.is_available():
        device = "cuda"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device_id = int(os.environ["LOCAL_RANK"])
            main_process = device_id == 0
        else:
            device_id = 0
            main_process = True
    else:
        device = "cpu"
        device_id = "cpu"
        device_count = 1
        main_process = True

    return device, device_id, device_count, main_process


def configure_optimizers(
    model: torch.nn.Module,
    weight_decay,
    learning_rate,
    betas,
    device_type,
):
    """Adapted from https://github.com/karpathy/nanoGPT"""
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"using fused AdamW: {use_fused}")

    return optimizer
