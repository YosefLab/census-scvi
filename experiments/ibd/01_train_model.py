from __future__ import annotations

import json
import logging
import os
import pathlib
import scvi
import sys
import torch
from inspect import signature

import anndata
import click
import lightning.pytorch as pl
from embedding_scvi import EmbeddingSCVI



def load_config(config_path: str) -> dict:
    """Load a JSON configuration file as a Python dictionary."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def make_parents(*paths) -> None:
    """Make parent directories of a file path if they do not exist."""
    for p in paths:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)


def wrap_kwargs(fn: callable) -> callable:
    """Wrap a function to accept keyword arguments from the command line."""
    for param in signature(fn).parameters:
        fn = click.option("--" + param, type=str)(fn)
    return click.command()(fn)


@wrap_kwargs
def main(config_path: str):
    config = load_config(config_path)
    adata_path = config["adata_path"]
    save_path = config["save_path"]
    layer = config["layer"]
    categorical_covariate_keys = config["categorical_covariate_keys"]
    experiment_name = config["experiment_name"]
    seed = config.get("seed", 2023)
    model_kwargs = config.get("model_kwargs", {})
    train_kwargs = config.get("train_kwargs", {})
    plan_kwargs = config.get("plan_kwargs", {})

    logging_dir = os.path.join(save_path, experiment_name)
    stdout_path = os.path.join(logging_dir, "stdout.log")
    stderr_path = os.path.join(logging_dir, "stderr.log")
    logs_path = os.path.join(logging_dir, "logs.log")
    make_parents(stdout_path, stderr_path, logs_path)
    stdout_handle = open(stdout_path, "w")
    stderr_handle = open(stderr_path, "w")
    sys.stdout = stdout_handle
    sys.stderr = stderr_handle
    logging.basicConfig(filename=logs_path, filemode="w")

    scvi.settings.seed = seed
    torch.set_float32_matmul_precision("medium")

    adata = anndata.read_h5ad(adata_path)
    EmbeddingSCVI.setup_anndata(
        adata,
        layer=layer,
        categorical_covariate_keys=categorical_covariate_keys
    )

    model = EmbeddingSCVI(adata, **model_kwargs)
    logger = pl.loggers.TensorBoardLogger(logging_dir)
    model.train(
        plan_kwargs=plan_kwargs,
        logger=logger,
        **train_kwargs,
    )

    model.save(
        os.path.join(logging_dir, "model"),
        overwrite=True,
        save_anndata=False,
    )


if __name__ == "__main__":
    main()
