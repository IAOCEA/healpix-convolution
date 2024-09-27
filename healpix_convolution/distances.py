import healpy as hp
import numpy as np

try:
    import dask.array as da

    dask_array_type = (da.Array,)
except ImportError:  # pragma: no cover
    da = None
    dask_array_type = ()


def cell_ids2vectors(cell_ids, nside, nest):
    flattened = cell_ids.flatten()
    vecs = np.stack(hp.pix2vec(nside, flattened, nest=nest), axis=-1)
    return np.reshape(vecs, cell_ids.shape + (3,))


def angle_between_vectors(a, b, axis):
    length_a = np.linalg.norm(a, axis=axis)
    length_b = np.linalg.norm(b, axis=axis)
    dot_product = np.sum(a * b, axis=axis)

    argument = np.clip(dot_product / (length_a * length_b), -1, 1)

    return np.arccos(argument)


def _distances(a, b, axis, nside, nest):
    vec_a = cell_ids2vectors(a, nside, nest)

    # TODO: contains `-1`, which `pix2vec` doesn't like
    mask = b != -1
    vec_b_ = cell_ids2vectors(np.where(mask, b, 0), nside, nest)
    vec_b = np.where(mask[..., None], vec_b_, np.nan)

    return angle_between_vectors(vec_a, vec_b, axis=axis)


def angular_distances(neighbours, *, resolution, indexing_scheme="nested", axis=None):
    """compute the angular great-circle distances between neighbours

    Parameters
    ----------
    neighbours : array-like
        The input cell ids.
    resolution : int
        The resolution of healpix pixelization.
    indexing_scheme : {"nested", "ring"}, default: "nested"
        The indexing scheme of the cell ids.
    axis : int, optional
        The axis used for the neighbours. If not given, assume the last dimension.

    Returns
    -------
    distances : array-like
        The great-circle distances in radians.
    """

    if axis is None:
        axis = -1

    nest = indexing_scheme == "nested"
    nside = 2**resolution

    if isinstance(neighbours, dask_array_type):
        return da.map_blocks(
            _distances,
            neighbours[:, :1],
            neighbours,
            axis=axis,
            nside=nside,
            nest=nest,
            chunks=neighbours.chunks,
        )
    else:
        return _distances(
            neighbours[:, :1], neighbours, axis=axis, nside=nside, nest=nest
        )
