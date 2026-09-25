import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import math
import numpy as np


TASK_NAMES = {
    "pcq": "PCQM4Mv2",
    "coco": "COCO-SP",
    "bridges": "Bridges",
    "flow": "Flow",
}

MODEL_NAME = {
    "lwl": "LWLT",
    "et": "ET",
    "gdt": "GDT",
}

TABLE_TEMPLATE = """\\begin{table}
    \centering
    \\resizebox{\\textwidth}{!}{TABULAR_STRING}
\caption{\\textbf{MODEL_NAME}: Optimal Hyperparameters, the maximum number of steps within the time budget, the average time per step, and the required GPU memory, on all tasks.}
\label{tab:MODEL_hps_stats}
\end{table}
"""


if __name__ == "__main__":

    total_table_string = ""
    for model in ["lwl", "et", "gdt"]:
        stats = []
        for task in TASK_NAMES.keys():
            try:
                leq32K = pd.read_csv(f"{model}-leq32K-{task}.csv")if os.path.exists(f"{model}-leq32K-{task}.csv") else None
                geq64K = pd.read_csv(f"{model}-geq64K-{task}.csv") if os.path.exists(f"{model}-geq64K-{task}.csv") else None
            except:
                print(model, task)
                raise ValueError("Error during CSV reading")
            if leq32K is None or geq64K is None:
                continue
            hps = pd.read_csv("selected_hyperparameters.csv")
            hps = hps[(hps["model"] == model.upper()) & (hps["task"] == task)]

            df = pd.concat([leq32K, geq64K])
            df = df[df["timing"] <= 36 * 3600]

            max_steps = int(df.groupby(["task"])["num_steps"].max().reset_index()["num_steps"].iloc[0])
            total_steps = int(df.groupby(["task"])["num_steps"].sum().reset_index()["num_steps"].iloc[0])
            total_time = float(df.groupby(["task"])["timing"].sum().reset_index()["timing"].iloc[0])
            memory = float(df.groupby(["task"])["memory"].mean().reset_index()["memory"].iloc[0])
            lr = float(hps["learning_rate"].iloc[0])
            ffn_dropout = float(hps["ffn_dropout"].iloc[0])

            stats.append({
                "Task": TASK_NAMES[task],
                "LR": f"{lr:.4f}",
                "MLP Dropout": f"{ffn_dropout:.1f}",
                "Max. \# Steps" : f"{max_steps}",
                "Avg. Time/Step": f"{(total_time / total_steps):.3f} s",
                "Avg. GPU Mem.": f"{(memory / 1024):.2f} GB"
            })

        table_string = TABLE_TEMPLATE.replace("TABULAR_STRING", pd.DataFrame(stats).to_latex(index=False))
        table_string = table_string.replace("MODEL_NAME", MODEL_NAME[model])
        table_string = table_string.replace("MODEL", model)
        total_table_string += table_string
        total_table_string += "\n\n\n\n"

    with open("plots/tab_stats.tex", "w") as f:
        f.write(total_table_string)
