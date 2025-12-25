import numpy as np

try:
    import dask.array as da

    dask_array_type = (da.Array,)
except ImportError:  # pragma: no cover
    da = None
    dask_array_type = ()


def spherical_to_euclidean(lon, lat):
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    return x, y, z


def cell_ids2vectors(cell_ids, grid_info):
    lon, lat = grid_info.cell_ids2geographic(cell_ids.flatten())
    vecs = np.stack(spherical_to_euclidean(np.deg2rad(lon), np.deg2rad(lat)), axis=-1)

    return np.reshape(vecs, cell_ids.shape + (3,))


def angle_between_vectors(a, b, axis):
    length_a = np.linalg.norm(a, axis=axis)
    length_b = np.linalg.norm(b, axis=axis)
    dot_product = np.sum(a * b, axis=axis)

    argument = np.clip(dot_product / (length_a * length_b), -1, 1)

    return np.arccos(argument)


def coord_between_vectors(a, b, axis):
    length_a = np.linalg.norm(a, axis=axis)
    length_b = np.linalg.norm(b, axis=axis)
    dot_product = np.sum(a - b, axis=axis)

    argument = np.clip(dot_product / (length_a * length_b), -1, 1)

    return np.arccos(argument)


def _distances(a, b, axis, grid_info):
    vec_a = cell_ids2vectors(a, grid_info)

    mask = b != -1
    vec_b_ = cell_ids2vectors(np.where(mask, b, 0), grid_info)
    vec_b = np.where(mask[..., None], vec_b_, np.nan)

    return angle_between_vectors(vec_a, vec_b, axis=axis)


def _coord(a, b, axis, grid_info, orientation=0):
    vec_a = cell_ids2vectors(a, grid_info)

    mask = b != -1
    vec_b_ = cell_ids2vectors(np.where(mask, b, 0), grid_info)
    vec_b = np.where(mask[..., None], vec_b_, np.nan)
    d = angle_between_vectors(vec_a, vec_b, axis=axis)
    dx = vec_b - vec_a
    theta = np.pi / 4
    X = vec_a[:, 0, 0] * np.cos(orientation * np.pi / 4) + vec_a[:, 0, 1] * np.sin(
        orientation * np.pi / 4
    )
    Y = -vec_a[:, 0, 0] * np.sin(orientation * np.pi / 4) + vec_a[:, 0, 1] * np.cos(
        orientation * np.pi / 4
    )
    Z = vec_a[:, 0, 2]
    dx[:, :, 0] = (vec_b - vec_a)[:, :, 0] * np.cos(orientation * np.pi / 4) + (
        vec_b - vec_a
    )[:, :, 1] * np.sin(orientation * np.pi / 4)
    dx[:, :, 1] = -(vec_b - vec_a)[:, :, 0] * np.sin(orientation * np.pi / 4) + (
        vec_b - vec_a
    )[:, :, 1] * np.cos(orientation * np.pi / 4)

    x = (
        dx[:, :, 1] * Z[:, None]
        + Y[:, None] * dx[:, :, 2]
        + dx[:, :, 1] * X[:, None] * (Z[:, None] >= 0)
        - dx[:, :, 1] * X[:, None] * (Z[:, None] < 0)
    )

    x = 0.5 * np.sqrt(x.shape[1]) * x / np.nanstd(x, 1)[:, None]

    return x, d * 2**grid_info.level


def angular_distances(neighbours, *, grid_info, axis=None):
    """compute the angular great-circle distances between neighbours

    Parameters
    ----------
    neighbours : array-like
        The input cell ids.
    grid_info : xdggs.HealpixInfo
        The grid parameters.
    axis : int, optional
        The axis used for the neighbours. If not given, assume the last dimension.

    Returns
    -------
    distances : array-like
        The great-circle distances in radians.
    """

    if axis is None:
        axis = -1

    if isinstance(neighbours, dask_array_type):
        return da.map_blocks(
            _distances,
            neighbours[:, :1],
            neighbours,
            axis=axis,
            grid_info=grid_info,
            chunks=neighbours.chunks,
        )
    else:
        return _distances(neighbours[:, :1], neighbours, axis=axis, grid_info=grid_info)


def coord_distances(neighbours, *, grid_info, axis=None, orientation=0):
    """compute the angular great-circle distances between neighbours

    Parameters
    ----------
    neighbours : array-like
        The input cell ids.
    grid_info : xdggs.HealpixInfo
        The grid parameters.
    axis : int, optional
        The axis used for the neighbours. If not given, assume the last dimension.

    Returns
    -------
    distances : array-like
        The great-circle distances in radians.
    """

    if axis is None:
        axis = -1

    if isinstance(neighbours, dask_array_type):
        return da.map_blocks(
            _distances,
            neighbours[:, :1],
            neighbours,
            axis=axis,
            grid_info=grid_info,
            chunks=neighbours.chunks,
        )
    else:
        return _coord(
            neighbours[:, :1],
            neighbours,
            axis=axis,
            grid_info=grid_info,
            orientation=orientation,
        )
