import xarray as xr


def convolve(ds, kernel, dim="cells"):
    """convolve data on a DGGS grid

    Parameters
    ----------
    ds : xarray.Dataset
        The input data. Must be compatible with `xdggs`.
    kernel : xarray.DataArray
        The sparse kernel matrix.
    dim : str, default: "cells"
        The input dimension name. Will also be the output dimension name.

    Returns
    -------
    convolved : xarray.Dataset
        The result of the convolution.
    """
    if ds.chunksizes:
        kernel = kernel.chunk()

    def _convolve(arr, weights):
        src_dims = ["input_cells"]

        if not set(src_dims).issubset(arr.dims):
            return arr

        return xr.dot(
            # drop all input coords, as those would most likely be broadcast
            arr.variable,
            weights,
            # This dimension will be "contracted"
            # or summed over after multiplying by the weights
            dims=src_dims,
        )

    return (
        ds.rename_dims({dim: "input_cells"})
        .map(_convolve, weights=kernel)
        .rename_dims({"output_cells": dim})
        .rename_vars({"output_cell_ids": "cell_ids"})
    )
