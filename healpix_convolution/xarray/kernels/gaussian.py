import xarray as xr
import xdggs  # noqa: F401

from healpix_convolution.kernels import gaussian


def compute_ring(grid_info, sigma, kernel_size, truncate):
    if kernel_size is not None:
        return int(kernel_size // 2)
    else:
        import healpy as hp

        cell_distance = hp.nside2resol(2**grid_info.resolution, arcmin=False)
        return int((truncate * sigma / cell_distance) // 2)


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
    grid_info = cell_ids.dggs.grid_info

    padded_cell_ids, matrix = gaussian.gaussian_kernel(
        cell_ids.data,
        resolution=grid_info.resolution,
        indexing_scheme=grid_info.indexing_scheme,
        sigma=sigma,
        truncate=truncate,
        kernel_size=kernel_size,
    )

    if kernel_size is not None:
        size_param = {"kernel_size": kernel_size}
    else:
        size_param = {"truncate": truncate}

    attrs = {
        "kernel_type": "gaussian",
        "method": "continuous",
        "sigma": sigma,
        "ring": compute_ring(
            kernel_size=kernel_size, grid_info=grid_info, truncate=truncate, sigma=sigma
        ),
    } | size_param

    return xr.DataArray(
        matrix,
        dims=["output_cells", "input_cells"],
        coords={
            "output_cell_ids": ("output_cells", cell_ids.data, cell_ids.attrs),
            "input_cell_ids": ("input_cells", padded_cell_ids, cell_ids.attrs),
        },
        attrs=attrs,
    )
