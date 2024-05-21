import healpy as hp
import numpy as np

from healpix_convolution.distances import distances
from healpix_convolution.kernels.common import create_sparse
from healpix_convolution.neighbours import neighbours


def gaussian_kernel(
    cell_ids,
    *,
    resolution: int,
    indexing_scheme: str,
    sigma: float,
    truncate: float = 4.0,
):
    cell_distance = hp.nside2resol(2**resolution, arcmin=False)
    ring = int((truncate * sigma / cell_distance) // 2)

    nb = neighbours(
        cell_ids, resolution=resolution, indexing_scheme=indexing_scheme, ring=ring
    )
    d = distances(nb, resolution=resolution, indexing_scheme=indexing_scheme)

    sigma2 = sigma * sigma
    phi_x = np.exp(-0.5 / sigma2 * d**2)

    shape = (12 * 4**resolution, 12 * 4**resolution)
    kernel = create_sparse(cell_ids, nb, phi_x, shape=shape)

    # normalize
    return kernel / np.sum(kernel, axis=1)
