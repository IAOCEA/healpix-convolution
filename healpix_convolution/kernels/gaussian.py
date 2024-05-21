import healpy as hp
import numpy as np

import healpix_convolution.distances as dist
import healpix_convolution.neighbours as nb

from .common import create_sparse


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

    neighbours = nb.neighbours(
        cell_ids, resolution=resolution, indexing_scheme=indexing_scheme, ring=ring
    )
    distances = dist.distances(
        neighbours, resolution=resolution, indexing_scheme=indexing_scheme
    )

    sigma2 = sigma * sigma
    phi_x = np.exp(-0.5 / sigma2 * distances**2)

    kernel = create_sparse(cell_ids, neighbours, phi_x)

    # normalize
    return kernel / np.sum(kernel, axis=1)
