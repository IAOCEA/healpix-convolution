import healpy as hp
import numpy as np

from healpix_convolution.distances import angular_distances
from healpix_convolution.kernels.common import create_sparse
from healpix_convolution.neighbours import neighbours


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
    resolution: int,
    indexing_scheme: str,
    sigma: float,
    truncate: float = 4.0,
    kernel_size: int | None = None,
):
    """construct a gaussian kernel on the healpix grid

    Parameters
    ----------
    cell_ids : array-like
        The cell ids.
    resolution : int
        The healpix resolution
    indexing_scheme : {"nested", "ring"}
        The healpix indexing scheme
    sigma : float
        The standard deviation of the gaussian kernel
    truncate : float, default: 4.0
        Truncate the kernel after this many multiples of ``sigma``.
    kernel_size : int, optional
        If given, determines the size of the kernel. In that case, ``truncate`` is ignored.

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

    # TODO: figure out whether there is a better way of defining the units of `sigma`
    if kernel_size is not None:
        ring = int(kernel_size / 2)
    else:
        cell_distance = hp.nside2resol(2**resolution, arcmin=False)
        ring = int((truncate * sigma / cell_distance) // 2)

    nb = neighbours(
        cell_ids, resolution=resolution, indexing_scheme=indexing_scheme, ring=ring
    )
    d = angular_distances(nb, resolution=resolution, indexing_scheme=indexing_scheme)
    weights = gaussian_function(d, sigma, mask=nb == -1)

    # TODO (keewis): figure out a way to translate global healpix indices to local ones
    # The kernel should still work for a subset of the full map.
    shape = (12 * 4**resolution, 12 * 4**resolution)
    return create_sparse(cell_ids, nb, weights, shape=shape)
