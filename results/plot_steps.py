import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import argparse
import os
import math

TEST_METRICS = {
    "pcq": "pcq_test_mae",
    "coco": "coco_test_f1",
    "bridges": "bridges_test_16_f1",
    "flow": "flow_test_16_mae",
}

TASK_TXT = {
    "pcq": "PCQM4M",
    "coco": "COCO-SP",
    "flow": "Flow",
    "bridges": "Bridges",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--ylog", action="store_true")
    parser.add_argument("--achieved", action="store_true")
    args = parser.parse_args()

    df = None

    models = ["lwl", "gdt"] if args.task == "coco" else ["lwl", "gdt", "et"]
    for model in models:
        leq32K = pd.read_csv(f"{model}-leq32K-{args.task}.csv")
        geq64K = pd.read_csv(f"{model}-geq64K-{args.task}.csv")

        _leq32K = leq32K.groupby(["task", "model", "model_size", "learning_rate", "ffn_dropout"])["best_val_score"].agg(
            ["mean", "std"]
        ).reset_index()

        if args.task in ["pcq", "flow"]:
            best = _leq32K.loc[_leq32K["mean"].idxmin()]
        else:
            best = _leq32K.loc[_leq32K["mean"].idxmax()]

        leq32K = leq32K[(leq32K["model_size"] == best["model_size"]) & (leq32K["learning_rate"] == best["learning_rate"]) & (leq32K["ffn_dropout"] == best["ffn_dropout"])]
        geq64K = geq64K[(geq64K["model_size"] == best["model_size"]) & (geq64K["learning_rate"] == best["learning_rate"]) & (geq64K["ffn_dropout"] == best["ffn_dropout"])]

        _df = pd.concat([leq32K, geq64K])
        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df])

    df["log_steps"] = df["num_steps"].apply(math.log)

    test_metric = TEST_METRICS[args.task]

    if args.ylog:
        df["log_" + test_metric] = df[test_metric].apply(math.log)
        test_metric = "log_" + test_metric

    sns.set_theme("notebook")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    y_label = "Test " + test_metric.split("_")[-1].upper()

    df["Model"] = df["model"].apply(lambda x: x if x != "LWL" else "LWLT")

    if not args.achieved:
        sns.lineplot(
            df,
            x="num_steps",
            y=test_metric,
            hue="Model",
            marker="o",
            ax=axs[0],
        )
        axs[0].set_xlabel("# Training steps")
    else:
        df["achieved_perfect_score_at"] = df["achieved_perfect_score_at"].fillna(df["num_steps"] - 1) + 1
        sns.scatterplot(
            df,
            x="achieved_perfect_score_at",
            y=test_metric,
            hue="Model",
            marker="o",
            ax=axs[0],
        )
        axs[0].set_xlabel("# Training steps")
    sns.scatterplot(
        df,
        x="timing",
        y=test_metric,
        hue="Model",
        marker="o",
        ax=axs[1],
    )

    axs[0].set_xscale("log")
    axs[0].xaxis.set_major_formatter(ScalarFormatter())
    axs[1].set_xscale("log")
    axs[1].xaxis.set_major_formatter(ScalarFormatter())

    axs[1].set_xlabel("Training time [s]")
    axs[0].set_title("Score vs. # of Steps", weight="bold")
    axs[1].set_title("Score vs. Training Time", weight="bold")
    axs[0].set_ylabel(y_label)
    axs[1].set_ylabel(y_label)
    fig.suptitle(TASK_TXT[args.task])

    plt.tight_layout()
    plt.savefig(f"plots/{args.task}_steps.pdf")
