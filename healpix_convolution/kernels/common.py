import numpy as np
import sparse


def create_sparse(cell_ids, neighbours, weights, shape):
    neighbours_ = np.reshape(neighbours, -1)
    mask = neighbours_ != -1

    n_neighbours = neighbours.shape[-1]
    cell_ids_ = np.reshape(
        np.repeat(cell_ids[..., None], repeats=n_neighbours, axis=-1), -1
    )

    coords = np.reshape(np.stack([cell_ids_, neighbours_], axis=0), (2, -1))

    weights_ = np.reshape(weights, -1)[mask]
    coords_ = coords[..., mask]

    return sparse.COO(coords=coords_, data=weights_, shape=shape)
