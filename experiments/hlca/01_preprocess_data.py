from __future__ import annotations

import os
from inspect import signature
from typing import Callable

import anndata
import click


def wrap_kwargs(fn: callable) -> callable:
    """Wrap a function to accept keyword arguments from the command line."""
    for param in signature(fn).parameters:
        fn = click.option("--" + param, type=str)(fn)
    return click.command()(fn)


@wrap_kwargs
def main(
    adata_path: str, save_path: str, core_or_extension: str = "core"
) -> anndata.AnnData:
    adata = anndata.read_h5ad(adata_path, backed="r")
    subset = adata[adata.obs["core_or_extension"] == core_or_extension]
    adata_out_path = os.path.join(save_path, f"hlca_{core_or_extension}.h5ad")
    subset.write_h5ad(adata_out_path, compression="gzip")
    return subset


if __name__ == "__main__":
    main()
