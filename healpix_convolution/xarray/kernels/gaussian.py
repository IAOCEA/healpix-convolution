import xarray as xr
import xdggs  # noqa: F401

from healpix_convolution.kernels import gaussian


def gaussian_kernel(
    cell_ids, sigma: float, truncate: float = 4.0, kernel_size: int | None = None
):
    """Create a symmetric gaussian kernel for the given cell ids

    Parameters
    ----------
    cell_ids : xarray.DataArray
        The cell ids. Must be valid according to xdggs.
    sigma : float
        The standard deviation of the gaussian function in radians.
    truncate : float, default: 4.0
        Truncate the kernel after this many multiples of sigma.
    kernel_size : int, optional
        If given, will be used instead of ``truncate`` to determine the size of the kernel.

    Returns
    -------
    kernel : xarray.DataArray
        The kernel as a sparse matrix.
    """
    dims = list(cell_ids.dims)

    grid = cell_ids.dggs.grid_info

    matrix = xr.apply_ufunc(
        gaussian.gaussian_kernel,
        cell_ids,
        kwargs={
            "resolution": grid.resolution,
            "indexing_scheme": grid.indexing_scheme,
            "sigma": sigma,
            "truncate": truncate,
            "kernel_size": kernel_size,
        },
        input_core_dims=[dims],
        output_core_dims=[["output_cells", "input_cells"]],
        dask="allowed",
        keep_attrs="drop",
    )

    if kernel_size is not None:
        size_param = {"kernel_size": kernel_size}
    else:
        size_param = {"truncate": truncate}

    return matrix.assign_attrs(
        {"kernel_type": "gaussian", "method": "continuous", "sigma": sigma} | size_param
    ).assign_coords(
        input_cell_ids=cell_ids.swap_dims({"cells": "input_cells"}).variable,
        output_cell_ids=cell_ids.swap_dims({"cells": "output_cells"}).variable,
    )
