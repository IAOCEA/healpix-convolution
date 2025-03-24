import healpy as hp
import numpy as np
import xdggs

from healpix_convolution.distances import coord_distances
from healpix_convolution.kernels.common import create_sparse, create_sparse_update
from healpix_convolution.neighbours import neighbours


def healpix_resolution(level):
    return np.sqrt(4 * np.pi / (12 * 4**level))


def compute_ring(level, sigma, truncate, kernel_size):
    if kernel_size is not None:
        ring = int(kernel_size // 2)
    else:
        cell_distance = healpix_resolution(level)
        ring = int((truncate * sigma / cell_distance) // 2)

    return ring if ring >= 1 else 1


def wavelet_function(x, distances, sigma, *, mask=None):

    phi_x = (np.cos(x) + 1j * np.sin(x)) * np.exp(-0.5 * distances**2)

    if mask is not None:
        masked = np.where(mask, 0, phi_x)
    else:
        masked = phi_x
    masked = masked - np.mean(masked, axis=1, keepdims=True)
    masked = masked / np.sqrt(np.nansum(abs(masked) ** 2, 1)[:, None])
    return masked


def wavelet_kernel(
    cell_ids,
    *,
    is_torch=False,
    grid_info: xdggs.HealpixInfo,
    orientation=0,
    truncate: float = 4.0,
    kernel_size: int | None = None,
    weights_threshold: float | None = None,
):
    """construct a gaussian kernel on the healpix grid

    Parameters
    ----------
    cell_ids : array-like
        The cell ids.
    grid_info : xdggs.HealpixInfo
        The grid parameters.
    sigma : float
        The standard deviation of the gaussian function in radians.
    truncate : float, default: 4.0
        Truncate the kernel after this many multiples of ``sigma``.
    kernel_size : int, optional
        If given, will be used instead of ``truncate`` to determine the size of the kernel.
    weights_threshold : float, optional
        If given, drop all kernel weights whose absolute value is smaller than this threshold.

    Returns
    -------
    kernel : sparse.COO
        The constructed kernel

    Notes
    -----
    no dask support, yet
    """
    if cell_ids.ndim != 1 or len([s for s in cell_ids.shape if s != 1]) != 1:
        raise ValueError(
            f"cell ids must be 1-dimensional, but shape is: {cell_ids.shape}"
        )

    cell_ids = np.reshape(cell_ids, (-1,))
    kernel_size = 5
    sigma = 1 / 2**grid_info.level
    ring = compute_ring(grid_info.level, sigma, truncate, kernel_size)

    nb = neighbours(cell_ids, grid_info=grid_info, ring=ring)

    x, d = coord_distances(nb, grid_info=grid_info, orientation=orientation)

    weights = wavelet_function(x, d, sigma, mask=nb == -1)

    return create_sparse(
        cell_ids, nb, weights, weights_threshold, is_torch=is_torch, is_complex=True
    )


def wavelet_smooth_kernel(
    cell_ids,
    *,
    is_torch=False,
    grid_info: xdggs.HealpixInfo,
    truncate: float = 4.0,
    kernel_size: int | None = None,
    weights_threshold: float | None = None,
):
    """construct a gaussian kernel on the healpix grid

    Parameters
    ----------
    cell_ids : array-like
        The cell ids.
    grid_info : xdggs.HealpixInfo
        The grid parameters.
    sigma : float
        The standard deviation of the gaussian function in radians.
    truncate : float, default: 4.0
        Truncate the kernel after this many multiples of ``sigma``.
    kernel_size : int, optional
        If given, will be used instead of ``truncate`` to determine the size of the kernel.
    weights_threshold : float, optional
        If given, drop all kernel weights whose absolute value is smaller than this threshold.

    Returns
    -------
    kernel : sparse.COO
        The constructed kernel

    Notes
    -----
    no dask support, yet
    """
    if cell_ids.ndim != 1 or len([s for s in cell_ids.shape if s != 1]) != 1:
        raise ValueError(
            f"cell ids must be 1-dimensional, but shape is: {cell_ids.shape}"
        )

    cell_ids = np.reshape(cell_ids, (-1,))
    kernel_size = 5
    sigma = 1 / 2**grid_info.level
    ring = compute_ring(grid_info.level, sigma, truncate, kernel_size)

    nb = neighbours(cell_ids, grid_info=grid_info, ring=ring)

    x, d = coord_distances(nb, grid_info=grid_info)
    weights = (abs(wavelet_function(x, d, sigma, mask=nb == -1))) ** 2
    weights /= np.sum(weights, 1)[:, None]
    return create_sparse(cell_ids, nb, weights, weights_threshold, is_torch=is_torch)


def wavelet_upgrade_kernel(
    cell_ids,
    *,
    grid_info: xdggs.HealpixInfo,
    truncate: float = 4.0,
    kernel_size: int | None = None,
    weights_threshold: float | None = None,
):
    """construct a gaussian kernel on the healpix grid

    Parameters
    ----------
    cell_ids : array-like
        The cell ids.
    grid_info : xdggs.HealpixInfo
        The grid parameters.
    sigma : float
        The standard deviation of the gaussian function in radians.
    truncate : float, default: 4.0
        Truncate the kernel after this many multiples of ``sigma``.
    kernel_size : int, optional
        If given, will be used instead of ``truncate`` to determine the size of the kernel.
    weights_threshold : float, optional
        If given, drop all kernel weights whose absolute value is smaller than this threshold.

    Returns
    -------
    kernel : sparse.COO
        The constructed kernel

    Notes
    -----
    no dask support, yet
    """
    if cell_ids.ndim != 1 or len([s for s in cell_ids.shape if s != 1]) != 1:
        raise ValueError(
            f"cell ids must be 1-dimensional, but shape is: {cell_ids.shape}"
        )

    cell_ids = np.reshape(cell_ids, (-1,))

    new_cell_ids = np.tile(cell_ids, 4) + np.repeat(np.arange(4), cell_ids.shape[0])

    t, p = hp.pix2ang(2 ** (grid_info.level + 1), new_cell_ids, nest=True)

    pix, weights = hp.get_interp_weights(2 ** (grid_info.level), t, p, nest=True)

    return create_sparse_update(
        np.tile(new_cell_ids, 4),
        pix.flatten(),
        weights.flatten(),
        2 ** (grid_info.level),
    )
