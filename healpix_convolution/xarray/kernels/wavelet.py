import xarray as xr
import xdggs  # noqa: F401

from healpix_convolution.kernels import wavelet
from healpix_convolution.kernels.wavelet import compute_ring


def wavelet_kernel(
    cell_ids,
    orientation=0,
    truncate: float = 4.0,
    kernel_size: int | None = None,
    weights_threshold: float | None = None,
):
    """Create a symmetric wavelet kernel for the given cell ids

    Parameters
    ----------
    cell_ids : xarray.DataArray
        The cell ids. Must be valid according to xdggs.
    sigma : float
        The standard deviation of the wavelet function in radians.
    truncate : float, default: 4.0
        Truncate the kernel after this many multiples of ``sigma``.
    kernel_size : int, optional
        If given, will be used instead of ``truncate`` to determine the size of the kernel.
    weights_threshold : float, optional
        If given, drop all kernel weights whose absolute value is smaller than this threshold.


    Returns
    -------
    kernel : xarray.DataArray
        The kernel as a sparse matrix.
    """
    grid_info = cell_ids.dggs.grid_info

    padded_cell_ids, matrix = wavelet.wavelet_kernel(
        cell_ids.data,
        orientation=orientation,
        grid_info=grid_info,
        truncate=truncate,
        kernel_size=kernel_size,
        weights_threshold=weights_threshold,
    )

    if kernel_size is not None:
        size_param = {"kernel_size": kernel_size}
    else:
        size_param = {"truncate": truncate}

    attrs = {
        "kernel_type": "gaussian",
        "method": "continuous",
        "sigma": 1 / 2**grid_info.level,
        "ring": compute_ring(
            grid_info.level,
            kernel_size=kernel_size,
            truncate=truncate,
            sigma=1 / 2**grid_info.level,
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


def wavelet_smooth_kernel(
    cell_ids,
    truncate: float = 4.0,
    kernel_size: int | None = None,
    weights_threshold: float | None = None,
):
    """Create a symmetric wavelet kernel for the given cell ids

    Parameters
    ----------
    cell_ids : xarray.DataArray
        The cell ids. Must be valid according to xdggs.
    sigma : float
        The standard deviation of the wavelet function in radians.
    truncate : float, default: 4.0
        Truncate the kernel after this many multiples of ``sigma``.
    kernel_size : int, optional
        If given, will be used instead of ``truncate`` to determine the size of the kernel.
    weights_threshold : float, optional
        If given, drop all kernel weights whose absolute value is smaller than this threshold.


    Returns
    -------
    kernel : xarray.DataArray
        The kernel as a sparse matrix.
    """
    grid_info = cell_ids.dggs.grid_info

    padded_cell_ids, matrix = wavelet.wavelet_smooth_kernel(
        cell_ids.data,
        grid_info=grid_info,
        truncate=truncate,
        kernel_size=kernel_size,
        weights_threshold=weights_threshold,
    )

    if kernel_size is not None:
        size_param = {"kernel_size": kernel_size}
    else:
        size_param = {"truncate": truncate}

    attrs = {
        "kernel_type": "gaussian",
        "method": "continuous",
        "sigma": 1 / 2**grid_info.level,
        "ring": compute_ring(
            grid_info.level,
            kernel_size=kernel_size,
            truncate=truncate,
            sigma=1 / 2**grid_info.level,
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


def wavelet_upgrade_kernel(
    cell_ids,
    truncate: float = 4.0,
    kernel_size: int | None = None,
    weights_threshold: float | None = None,
):
    """Create a symmetric wavelet kernel for the given cell ids

    Parameters
    ----------
    cell_ids : xarray.DataArray
        The cell ids. Must be valid according to xdggs.
    sigma : float
        The standard deviation of the wavelet function in radians.
    truncate : float, default: 4.0
        Truncate the kernel after this many multiples of ``sigma``.
    kernel_size : int, optional
        If given, will be used instead of ``truncate`` to determine the size of the kernel.
    weights_threshold : float, optional
        If given, drop all kernel weights whose absolute value is smaller than this threshold.


    Returns
    -------
    kernel : xarray.DataArray
        The kernel as a sparse matrix.
    """
    grid_info = cell_ids.dggs.grid_info

    padded_cell_ids, matrix = wavelet.wavelet_upgrade_kernel(
        cell_ids.data,
        grid_info=grid_info,
        truncate=truncate,
        kernel_size=kernel_size,
        weights_threshold=weights_threshold,
    )

    if kernel_size is not None:
        size_param = {"kernel_size": kernel_size}
    else:
        size_param = {"truncate": truncate}

    attrs = {
        "kernel_type": "gaussian",
        "method": "continuous",
        "sigma": 1 / 2**grid_info.level,
        "ring": compute_ring(
            grid_info.level,
            kernel_size=kernel_size,
            truncate=truncate,
            sigma=1 / 2**grid_info.level,
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
