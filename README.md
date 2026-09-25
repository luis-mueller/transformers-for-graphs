
# Transformers for Graphs

## Install
- Python version `3.10.19`
- Install the packages as listed in `packages.txt`

I recommend you use a virtual environment.

## Running the Experiments
The experiments in the thesis were run in three distinct stages:

1. Preprocessing: See `scripts/preprocessing` for the preprocessing scripts used for the thesis.
2. LEQ32K: See `scripts/[model]-leq32K` for the training runs using 32768 or fewer steps. Then, I ran `results/compute_selected_hyperparamters.py` to compute the hyperparameters for the GEQ64K runs.
3. GEQ64K: See `scripts/[model]-geq64K` for the training runs using 65536 or more steps.

The `results` folder contains additional helper scripts for plotting and computing the training statistics.
