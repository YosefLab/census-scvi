from __future__ import annotations

import os
import pathlib
import sys
from inspect import signature
from typing import Callable

import anndata
import click
import scvi
import torch
from ray import tune


torch.set_float32_matmul_precision("medium")

SEARCH_SPACE = {
    "n_latent": tune.choice([50, 100, 200]),
    "n_layers": tune.choice([1, 2, 5]),
}

MODEL_KWARGS = {
    "n_hidden": 512,
    "var_activation": torch.nn.functional.softplus
}

TRAIN_KWARGS = {
    "plan_kwargs": {
        "lr": 1e-4,
    },
    "batch_size": 1024
}

SCHEDULER_KWARGS = {
    "grace_period": 5,
}


def wrap_kwargs(fn: Callable) -> Callable:
    """Wrap a function to accept keyword arguments from the command line."""
    for param in signature(fn).parameters:
        fn = click.option("--" + param, type=str)(fn)
    return click.command()(fn)


def make_parents(*paths) -> None:
    """Make parent directories of a file path if they do not exist."""
    for p in paths:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)


def load_anndata(adata_path: str) -> anndata.AnnData:
    """Load AnnData into memory."""
    return anndata.read_h5ad(adata_path)


def setup_anndata(adata: anndata.AnnData, batch_key: str) -> None:
    """Setup AnnData for scVI."""
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)


def setup_tuner() -> scvi.autotune.ModelTuner:
    """Setup autotune with scVI."""
    return scvi.autotune.ModelTuner(scvi.model.SCVI)


def fit_tuner(
    tuner: scvi.autotune.ModelTuner, 
    adata: anndata.AnnData,
    search_space: dict,
    model_kwargs: dict,
    train_kwargs: dict,
    num_cpus: int,
    num_gpus: int,
    seed: int | None,
    experiment_name: str,
    save_dir: str,
    num_samples: int = 10,
    max_epochs: int = 20,
    scheduler: str = "asha",
    searcher: str = "hyperopt",
    scheduler_kwargs: dict = SCHEDULER_KWARGS,
):
    return tuner.fit(
        adata,
        metric="validation_loss",
        search_space=search_space,
        model_kwargs=model_kwargs,
        train_kwargs=train_kwargs,
        use_defaults=False,
        num_samples=num_samples,
        max_epochs=max_epochs,
        scheduler=scheduler,
        searcher=searcher,
        scheduler_kwargs=scheduler_kwargs,
        resources={"cpu": num_cpus, "gpu": num_gpus},
        seed=seed,
        experiment_name=experiment_name,
        logging_dir=save_dir,
    )


@wrap_kwargs
def main(
    adata_path: str,
    batch_key: str,
    num_cpus: str,
    num_gpus: str,
    seed: int | None = 2023,
    experiment_name: str = "autotune_scvi_v2",
    save_dir: str = "/data",
):
    logging_dir = os.path.join(save_dir, experiment_name)
    stdout_path = os.path.join(logging_dir, "stdout.log")
    stderr_path = os.path.join(logging_dir, "stderr.log")
    make_parents(stdout_path, stderr_path)
    stdout_handle = open(stdout_path, "w")
    stderr_handle = open(stderr_path, "w")
    sys.stdout = stdout_handle
    sys.stderr = stderr_handle

    adata = load_anndata(adata_path)
    setup_anndata(adata, batch_key)
    tuner = setup_tuner()
    fit_tuner(
        tuner, 
        adata, 
        SEARCH_SPACE,
        MODEL_KWARGS,
        TRAIN_KWARGS,
        float(num_cpus),
        float(num_gpus),
        seed,
        experiment_name,
        save_dir,
    )

    stdout_handle.close()
    stderr_handle.close()

if __name__ == "__main__":
    main()
