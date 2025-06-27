import xarray as xr

from healpix_convolution.xarray.padding import pad


def convolve(
    ds, kernel, *, dim="cells", mode: str = "constant", constant_value: int | float = 0
):
    """convolve data on a DGGS grid

    Parameters
    ----------
    ds : xarray.Dataset
        The input data. Must be compatible with `xdggs`.
    kernel : xarray.DataArray
        The sparse kernel matrix.
    dim : str, default: "cells"
        The input dimension name. Will also be the output dimension name.
    mode : str, default: "constant"
        Mode used to pad the array before convolving. Only used for regional maps. See
        :py:func:`pad` for a list of available modes.
    constant_values : int or float, default: 0
        The constant value to pad with when mode is ``"constant"``.

    Returns
    -------
    convolved : xarray.Dataset
        The result of the convolution.

    See Also
    --------
    healpix_convolution.xarray.pad
    healpix_convolution.pad
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

    if ds.sizes["cells"] != kernel.sizes["input_cells"]:
        padder = pad(
            ds["cell_ids"],
            ring=kernel.attrs["ring"],
            mode=mode,
            constant_value=constant_value,
        )
        ds = padder.apply(ds)

    unrelated = ds.drop_vars(
        [name for name, var in ds.variables.items() if "cells" in var.dims]
    )

    return (
        ds.rename_dims({dim: "input_cells"})
        .map(_convolve, weights=kernel)
        .rename_dims({"output_cells": dim})
        .rename_vars({"output_cell_ids": "cell_ids"})
        .merge(unrelated)
        .dggs.decode()
    )
