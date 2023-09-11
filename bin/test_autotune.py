import pathlib
from inspect import signature
from typing import Callable

import anndata
import click
import scvi
import torch
from ray import tune

torch.set_float32_matmul_precision("medium")

SEARCH_SPACE = {
    "n_hidden": tune.choice([128, 256, 512]),
    "n_layers": tune.choice([1, 10, 25]),
    "n_latent": tune.choice([10, 50, 100]),
    "lr": tune.choice([1e-2, 1e-3, 1e-4]),
    "batch_size": tune.choice([512, 1024, 2048]),
    # hyperparams below are not being tuned
    "var_activation": tune.choice([torch.nn.functional.softplus])  # nans
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


def load_sample_anndata() -> anndata.AnnData:
    return scvi.data.synthetic_iid()


def setup_anndata(adata: anndata.AnnData) -> None:
    scvi.model.SCVI.setup_anndata(adata)


def setup_tuner() -> scvi.autotune.ModelTuner:
    return scvi.autotune.ModelTuner(scvi.model.SCVI)


def fit_tuner(
    tuner: scvi.autotune.ModelTuner, 
    adata: anndata.AnnData,
    search_space: dict,
):
    return tuner.fit(
        adata,
        metric="validation_loss",
        search_space=search_space,
        resources={"cpu": 1, "gpu": 1},
        experiment_name="test_autotune",
        logging_dir="/data/autotune_logs",
    )


def main():
    adata = load_sample_anndata()
    setup_anndata(adata)
    tuner = setup_tuner()
    fit_tuner(tuner, adata, SEARCH_SPACE)


if __name__ == "__main__":
    main()
