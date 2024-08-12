import dask
import dask.array as da
import numpy as np
import sparse


def create_sparse(cell_ids, neighbours, weights, value_threshold):
    neighbours_ = np.reshape(neighbours, (-1,))

    all_cell_ids = np.unique(neighbours_)
    if all_cell_ids[0] == -1:
        all_cell_ids = all_cell_ids[1:]

    row_indices = np.reshape(
        np.broadcast_to(np.arange(cell_ids.size)[:, None], shape=neighbours.shape),
        (-1,),
    )
    column_indices = np.searchsorted(all_cell_ids, neighbours_, side="left")

    coords = np.stack([row_indices, column_indices], axis=0)

    mask = (neighbours_ != -1) & (np.abs(weights) >= value_threshold)

    weights_ = np.reshape(weights, (-1,))[mask]
    coords_ = coords[..., mask]

    if isinstance(weights_, da.Array):
        coords_, weights_ = dask.compute(coords_, weights_)

    shape = (cell_ids.size, all_cell_ids.size)
    return all_cell_ids, sparse.COO(
        coords=coords_, data=weights_, shape=shape, fill_value=0
    )
