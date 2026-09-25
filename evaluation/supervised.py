import torch


@torch.inference_mode()
def evaluate_supervised(task, loaders, funcs, model, split, ctx, device):
    metrics = {}
    y_true = []
    y_pred = []
    metric_name, metric_func = funcs["metric"]

    for batch in loaders[split]:
        batch = batch.to(device)

        with ctx:
            preds = model(batch)
        y = batch.y
        y = [y.detach().cpu()]
        preds = [preds.detach().cpu()]

        y_true += y
        y_pred += preds
    metrics[f"{task}_{split}_{metric_name}"] = metric_func(y_pred, y_true)
    return metrics
