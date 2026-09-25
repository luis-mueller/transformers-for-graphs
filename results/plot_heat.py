import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import math
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm


TEST_METRICS = {
    "bridges": "bridges_test_{0}_f1",
    "flow": "flow_test_{0}_mae",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--log_norm", action="store_true")
    args = parser.parse_args()

    df = None

    for model in ["lwl", "gdt"] if args.task == "coco" else ["lwl", "gdt", "et"]:
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

    ood_cols = [TEST_METRICS[args.task].format(20 + i * 4) for i in range(12)]
    df = df[["model", "num_steps"] + ood_cols]

    for i, col in enumerate(ood_cols):
        df[str(20 + i * 4)] = df[col]
        df = df.drop(col, axis=1)
    df = df.melt(
        id_vars=["model", "num_steps"],
        var_name="test_size",
        value_name=TEST_METRICS[args.task].format(0)
    )

    vmin = df[TEST_METRICS[args.task].format(0)].min()
    vmax = df[TEST_METRICS[args.task].format(0)].max()
    full_cols = df["num_steps"].unique()

    sns.set_theme("paper")
    fig, axs = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True, width_ratios=[0.8, 1])
    change_name = lambda x: x if x != "lwl" else "lwlt"
    for i, model in enumerate(["lwl", "gdt", "et"]):
        df["model"] = df["model"].apply(change_name)
        data = df[df["model"] == model.upper()]
        mean_tab = data.pivot_table(index="test_size", columns="num_steps", values=TEST_METRICS[args.task].format(0), aggfunc="mean").reindex(columns=full_cols)
        std_tab = data.pivot_table(index="test_size", columns="num_steps", values=TEST_METRICS[args.task].format(0), aggfunc="std").reindex(columns=full_cols)
        
        sns.heatmap(
            mean_tab,
            annot=True, 
            ax=axs[i][0], 
            square=False, 
            cbar=False,
            norm=LogNorm(vmin, vmax) if args.log_norm else None,
            vmin=vmin,
            vmax=vmax,
            cmap="crest",
            robust=True,
            fmt=".2f"
        )

        sns.heatmap(
            std_tab, 
            annot=True,
            ax=axs[i][1], 
            square=False, 
            cbar=True,
            norm=LogNorm(vmin, vmax) if args.log_norm else None,
            vmin=vmin,
            vmax=vmax,
            cmap="crest",
            robust=True,
            fmt=".2f"
        )

        axs[i][0].set_ylabel("$\\bf{" + change_name(model).upper() + "}$" + "\nGraph size")
        axs[i][0].set_xlabel("# Training steps")
        axs[i][1].set_ylabel('')
        axs[i][1].set_xlabel("# Training steps")
        if i < 2:
            axs[i][0].set_xlabel('')
            axs[i][1].set_xlabel('')
        
        if i == 0:
            axs[i][0].set_title("Test " + TEST_METRICS[args.task].split("_")[-1].upper())
            axs[i][1].set_title("Standard Deviation")

    plt.tight_layout()
    plt.savefig(f"plots/{args.task}_heat.pdf")
