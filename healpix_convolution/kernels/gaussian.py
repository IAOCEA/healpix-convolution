import numpy as np
import xdggs

from healpix_convolution.distances import angular_distances
from healpix_convolution.kernels.common import create_sparse
from healpix_convolution.neighbours import neighbours


def healpix_resolution(level):
    return np.sqrt(4 * np.pi / 12 * 4**level)


def compute_ring(resolution, sigma, truncate, kernel_size):
    if kernel_size is not None:
        ring = int(kernel_size // 2)
    else:
        cell_distance = healpix_resolution(resolution)
        ring = int((truncate * sigma / cell_distance) // 2)

    return ring if ring >= 1 else 1


def gaussian_function(distances, sigma, *, mask=None):
    sigma2 = sigma * sigma
    phi_x = np.exp(-0.5 / sigma2 * distances**2)

    if mask is not None:
        masked = np.where(mask, 0, phi_x)
    else:
        masked = phi_x

    return masked / np.sum(masked, axis=1, keepdims=True)


def gaussian_kernel(
    cell_ids,
    *,
    grid_info: xdggs.HealpixInfo,
    sigma: float,
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

    ring = compute_ring(grid_info.level, sigma, truncate, kernel_size)

    nb = neighbours(cell_ids, grid_info=grid_info, ring=ring)
    d = angular_distances(nb, grid_info=grid_info)
    weights = gaussian_function(d, sigma, mask=nb == -1)

    return create_sparse(cell_ids, nb, weights, weights_threshold)
