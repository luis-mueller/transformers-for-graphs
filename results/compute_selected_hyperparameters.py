import pandas as pd
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    data = pd.read_csv(f"{args.model}-leq32K-{args.task}.csv")
    df = data.groupby(["task", "model", "model_size", "learning_rate", "ffn_dropout"])["best_val_score"].agg(
        ["mean", "std"]
    ).reset_index()
    print(df)
    if args.task in ["coco", "bridges"]:
        df = pd.DataFrame([df.loc[df["mean"].idxmax()]])
    else:
        df = pd.DataFrame([df.loc[df["mean"].idxmin()]])
    if os.path.exists("selected_hyperparameters.csv"):
        df.to_csv("selected_hyperparameters.csv", header=False, mode="a", index=False)
    else:
        df.to_csv("selected_hyperparameters.csv", header=True, index=False)