import xarray as xr

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

    matrix = xr.apply_ufunc(
        gaussian.gaussian_kernel,
        cell_ids,
        kwargs={
            "resolution": cell_ids.xdggs.params["resolution"],
            "indexing_scheme": cell_ids.xdggs.params["indexing_scheme"],
            "sigma": sigma,
            "truncate": truncate,
            "kernel_size": kernel_size,
        },
        input_core_dims=[list(cell_ids.dims)],
        output_core_dims=[["convolved", dims[0]]],
        dask="forbidden",  # until `gaussian_kernel` supports it
        keep_attrs="drop",
    )

    if kernel_size is not None:
        size_param = {"kernel_size": kernel_size}
    else:
        size_param = {"truncate": truncate}

    return matrix.assign_attrs(
        cell_ids.xdggs.params
        | {"kernel_type": "gaussian", "method": "continous", "sigma": sigma}
        | size_param
    )
