import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def get_trial_dirs(
    root_dir: str, 
    filter_incomplete: bool = False,
    filter_lightning: bool = False,
) -> list[str]:
    trial_dirs = [
        trial for trial in os.listdir(root_dir) if trial.startswith("_trainable")
    ]
    if filter_lightning:
        trial_dirs = [
            trial for trial in trial_dirs if "lightning" not in trial
        ]
    if not filter_incomplete:
        return trial_dirs
    
    for trial_dir in trial_dirs:
        if not os.path.exists(os.path.join(root_dir, trial_dir, "progress.csv")):
            print(f"Missing progress.csv for {trial_dir}")
            trial_dirs.remove(trial_dir)
            continue
        if not os.path.exists(os.path.join(root_dir, trial_dir, "params.json")):
            print(f"Missing params.json for {trial_dir}")
            trial_dirs.remove(trial_dir)
            continue

    return trial_dirs


def get_trial_ids(trial_dirs: list[str]) -> list[str]:
    return [trial_dir.split("_")[3] for trial_dir in trial_dirs]


def get_trial_params(
    root_dir: str, trial_dir: str, params: list[str]
) -> dict[str, str]:
    """Get the model and training parameters for a trial."""
    trial_dir = os.path.join(root_dir, trial_dir)
    params_path = os.path.join(trial_dir, "params.json")
    with open(params_path, "r") as f:
        trial_params = json.load(f)

    return {param: trial_params[param] for param in params}


def get_trial_progress(root_dir: str, trial_dir: str) -> pd.DataFrame:
    """Get the training progress for a trial."""
    trial_dir = os.path.join(root_dir, trial_dir)
    progress_path = os.path.join(trial_dir, "progress.csv")
    return pd.read_csv(progress_path)


def get_trial_info(
    trial_params: dict[str, dict], trial_progress: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Get the model and training parameters for all trials."""
    trial_info = {}
    for trial_id, progress in trial_progress.items():
        params = trial_params[trial_id]
        trial_iters = len(progress)
        trial_info[trial_id] = {**params, "iters": trial_iters}

    return pd.DataFrame.from_dict(trial_info, orient="index")


def plot_trials(
    trial_info: pd.DataFrame,
    trial_progress: dict[str, pd.DataFrame],
    trial_params: dict[str, dict],
    param: str,
    metric: str = "validation_loss",
    iters: int | None = None,
    figsize: tuple[float, float] = (12, 8),
    ylim: tuple[float, float] | None = None,
):
    if iters is not None:
        trial_info = trial_info[trial_info["iters"] == iters]

    plt.figure(figsize=figsize)
    plt.title(f"{metric} by {param}")
    if ylim is not None:
        plt.ylim(*ylim)

    values = trial_info[param].unique()
    values_to_colors = {value: f"C{i}" for i, value in enumerate(values)}

    for trial_id in trial_info.index:
        progress = trial_progress[trial_id]
        params = trial_params[trial_id]
        label = f"{param}={params[param]}"
        plt.plot(
            progress[metric],
            label=label,
            marker="o",
            color=values_to_colors[params[param]],
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
    )
    
    plt.xlabel("Iterations")
    plt.ylabel(metric)
    plt.show()
    