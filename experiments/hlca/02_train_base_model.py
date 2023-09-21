from __future__ import annotations

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
def main(
    adata_path: str,
    save_path: str,
    categorical_covariate_keys: list[str],
    experiment_name: str,
    seed: int,
    max_epochs: int,
):
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

    categorical_covariate_keys = categorical_covariate_keys.split(",")
    seed = int(seed)
    max_epochs = int(max_epochs)
    scvi.settings.seed = seed
    torch.set_float32_matmul_precision("medium")

    adata = anndata.read_h5ad(adata_path)
    EmbeddingSCVI.setup_anndata(
        adata, 
        categorical_covariate_keys=categorical_covariate_keys
    )

    model = EmbeddingSCVI(adata)
    logger = pl.loggers.TensorBoardLogger(logging_dir)
    model.train(
        max_epochs=max_epochs,
        batch_size=1024,
        load_sparse_tensor=True,
        early_stopping=True,
        early_stopping_patience=5,
        check_val_every_n_epoch=1,
        logger=logger,
        plan_kwargs={
            "lr": 1e-4,
            "reduce_lr_on_plateau": True,
            "n_epochs_kl_warmup": 0,
            "min_kl_weight": 0.25,
            "max_kl_weight": 0.25,
        },
    )

    model.save(
        os.path.join(logging_dir, "model"),
        overwrite=True,
        save_anndata=False,
    )


if __name__ == "__main__":
    main()
