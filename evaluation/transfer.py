import torch
import tqdm
import math
from loguru import logger


@torch.inference_mode()
def evaluate_transfer(task, loaders, funcs, model, split, ctx, device, num_samples, num_topk=5):
    metrics = {}
    metric_name, metric_func = funcs["metric"]

    batch_size = loaders["train"].loader.batch_size
    num_batches = math.ceil(num_samples / batch_size)
    logger.info(f"Collecting {num_samples} train embeddings from {num_batches} batches.")
    train_set = []
    y = []
    for _ in tqdm.tqdm(range(num_batches)):
        batch1 = loaders["train"].sample().to(device)
        with torch.inference_mode():
            with ctx:
                train = model(batch1, get_embeddings=True)
                train_set.append(train)
                y.append(batch1.y)

    train_set = torch.cat(train_set)[:num_samples]
    y = torch.cat(y)

    logger.info("Computing query embeddings and distances to train embeddings")
    y_true = []
    y_pred = []
    for batch2 in tqdm.tqdm(loaders["test"]):
        batch2 = batch2.to(device)
        with torch.inference_mode():
            with ctx:
                query = model(batch2, get_embeddings=True)

        num_processed = 0
        to_process = 1000
        dists = []
        while num_processed < train_set.size(0):
            train = train_set[num_processed : num_processed + to_process]
            dist = torch.linalg.norm((query[:, None] - train), dim=2)
            dists.append(dist)
            num_processed += to_process

        dist = torch.cat(dists, 1)
        idx = dist.topk(num_topk, largest=False).indices
        preds = y[idx].mode(1).values

        y_true.append(batch2.y.detach().cpu())
        y_pred.append(preds.detach().cpu())

    score = metric_func(y_pred, y_true)
    metrics[f"{task}_transfer_{metric_name}"] = score
    return metrics