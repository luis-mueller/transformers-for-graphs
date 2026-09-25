import os
import torch.nn as nn
import pandas as pd
from loguru import logger
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tasks.vision import load_coco, load_pascal
from tasks.algo_reaso import load_flow, load_bridges, load_mst, load_cycles
from tasks.molecules import load_pcqm4mv2

TASKS = {
    "pascal": load_pascal,
    "coco": load_coco,
    "pcq": load_pcqm4mv2,
    "bridges": load_bridges,
    "mst": load_mst,
    "flow": load_flow,
    "cycles": load_cycles,
}

TASK_LEVELS = {
    "pascal": "node",
    "coco": "node",
    "pcq": "graph",
    "code": "seq",
    "bridges": "edge",
    "mst": "edge",
    "flow": "graph",
    "cycles": "node",
}


TASK_BATCH_SIZE = {
    "pcq": 256,
    "coco": 32,
    "code": 32,
    "bridges": 256,
    "mst": 256,
    "flow": 256,
    "cycles": 1,
    "pascal": 1,
}


class TrainingDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, device_count=1):
        if device_count > 1:
            self.loader = DataLoader(
                dataset,
                batch_size=batch_size // device_count,
                shuffle=False,
                sampler=DistributedSampler(dataset),
            )
            self.loader.sampler.set_epoch(0)
        else:
            self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.iterator = iter(self.loader)
        self.device_count = device_count
        self.epoch = 0

    def sample(self):
        minibatch = next(self.iterator, None)
        if minibatch is None:
            self.iterator = iter(self.loader)
            minibatch = next(self.iterator, None)
            if self.device_count > 1:
                self.epoch += 1
                self.loader.sampler.set_epoch(self.epoch)
        return minibatch


def extract_task(
    batch_size, test_batch_size, datasets, modules, funcs, device_count, main_process
):
    if test_batch_size is None:
        test_batch_size = batch_size
    return (
        {
            "train": TrainingDataLoader(
                datasets["train"], batch_size, device_count=device_count
            ),
            "valid": (
                DataLoader(datasets["valid"], batch_size // device_count)
                if main_process
                else None
            ),
            **{
                test_split: (
                    DataLoader(datasets[test_split], test_batch_size // device_count)
                    if main_process
                    else None
                )
                for test_split in [k for k in datasets.keys() if k.startswith("test")]
            },
        },
        nn.ModuleDict(modules),
        funcs,
    )


def add_task_args(parser):
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--task", type=str, choices=TASKS.keys())
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument(
        "--hour_budget",
        type=int,
        default=1,
        help="Number of hours for the run",
    )
    parser.add_argument(
        "--timing_lookup_path",
        type=str,
        default="./results/timing_lookup.csv",
        help="Path to a lookup table for model timings to adjust number of training steps",
    )


def load_task(
    task,
    root,
    embed_dim=1,
    bias=False,
    pre_transform=None,
    transform=None,
    batch_size=None,
    test_batch_size=None,
    device_count=1,
    main_process=True,
):
    _load = TASKS[task]
    logger.info(f"Loading task {task}")
    task_data = _load(
        root=os.path.join(root, task),
        embed_dim=embed_dim,
        bias=bias,
        pre_transform=pre_transform,
        transform=transform,
    )

    if batch_size is not None:
        _data, _modules, _funcs = extract_task(
            batch_size,
            test_batch_size,
            *task_data,
            device_count=device_count,
            main_process=main_process,
        )
        _level = TASK_LEVELS[task]
        return _data, _modules, _funcs, _level


def load_task_from_args(args, embed_dim, bias, transform, device_count, main_process):
    return load_task(
        args.task,
        args.root,
        embed_dim,
        bias,
        transform=transform,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        device_count=device_count,
        main_process=main_process,
    )


def get_steps_per_hour(timing_lookup_path, task, model, model_size):
    steps_per_h = None
    if os.path.exists(timing_lookup_path):
        data = pd.read_csv(timing_lookup_path)
        data = data[
            (data["task"] == task)
            & (data["model"] == model)
            & (data["model_size"] == model_size)
        ]
        if len(data) > 0:
            steps_per_s = data.iloc[0]["mean"]
            steps_per_h = steps_per_s * 3600
            logger.info(f"Determined hour rate of {steps_per_h:.4f} steps/h")
    return steps_per_h


def override_runtime_behavior(args):
    args.batch_size = TASK_BATCH_SIZE[args.task]
    args.warmup_iters = int(args.num_steps * 0.025)

    if args.task in ["coco", "bridges"]:
        args.checkpoint = os.path.join(args.root, f"checkpoints/{args.task}-{args.model}-{args.model_size}-{args.learning_rate}-{args.ffn_dropout}-{args.num_steps}-{args.seed}.pt")

    return args
