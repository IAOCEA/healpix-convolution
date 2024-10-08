import healpy as hp
import numba
import numpy as np
from numba import int8, int16, int32, int64

try:
    import dask.array as da

    dask_array_type = (da.Array,)
except ImportError:  # pragma: no cover
    dask_array_type = ()
    da = None

face_neighbours = np.array(
    [
        [8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9],
        [5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8],
        [-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1],
        [4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4],
        [-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1],
        [3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7],
        [2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3],
    ],
    dtype="int8",
)
swap_arrays = np.array(
    [
        [0, 0, 3],
        [0, 0, 6],
        [0, 0, 0],
        [0, 0, 5],
        [0, 0, 0],
        [5, 0, 0],
        [0, 0, 0],
        [6, 0, 0],
        [3, 0, 0],
    ],
    dtype="uint8",
)


def cell_ids2xyf(cell_ids, *, nside, indexing_scheme):
    nest_values = {"nested": True, "ring": False}
    nest = nest_values.get(indexing_scheme)
    if nest is None:
        raise ValueError(f"unsupported indexing scheme: {indexing_scheme}")

    return hp.pixelfunc.pix2xyf(nside, cell_ids, nest=nest)


def xyf2cell_ids(x, y, face, *, nside, indexing_scheme):
    nest_values = {"nested": True, "ring": False}
    nest = nest_values.get(indexing_scheme)
    if nest is None:
        raise ValueError(f"unsupported indexing scheme: {indexing_scheme}")

    return hp.pixelfunc.xyf2pix(nside, x, y, face, nest=nest)


def generate_offsets(ring):
    steps = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 1)]
    cx = 0
    cy = 0

    for c in range(ring + 1):
        # 1: go c steps to the left and yield
        # 2: turn right
        # 3: go 1 step at a time while yielding, until we reach the maximum distance
        # 4: repeat 2 and 3 until we turned 4 times
        # 5: go up until just before we reached the first point
        x = cx - c
        y = cy - 0

        yield x, y

        if c == 0:
            continue

        for index, (sx, sy) in enumerate(steps):
            if index == 0:
                n_steps = c
            elif index == 4:
                n_steps = c - 1
            else:
                n_steps = 2 * c

            for _ in range(n_steps):
                x = x + sx
                y = y + sy

                yield (x, y)


@numba.njit
def adjust_xyf(cx, cy, cf, nside):
    if (cx >= 0 and cx < nside) and (cy >= 0 and cy < nside):
        return cx, cy, cf

    nbnum = 4
    if cx < 0:
        cx = cx + nside
        nbnum -= 1
    elif cx >= nside:
        cx = cx - nside
        nbnum += 1

    if cy < 0:
        cy = cy + nside
        nbnum -= 3
    elif cy >= nside:
        cy = cy - nside
        nbnum += 3

    nf = face_neighbours[nbnum][cf]
    if nf < 0:
        # invalid pixel
        return -1, -1, -1

    bits = swap_arrays[nbnum][cf >> 2]
    if bits & 1:
        nx = nside - cx - 1
    else:
        nx = cx
    if bits & 2:
        ny = nside - cy - 1
    else:
        ny = cy

    if bits & 4:
        nx, ny = ny, nx

    return nx, ny, nf


@numba.guvectorize(
    [
        (int32, int32, int32, int32, int8[:, :], int32[:], int32[:], int32[:]),
        (int64, int64, int64, int32, int8[:, :], int64[:], int64[:], int64[:]),
        (int32, int32, int32, int32, int16[:, :], int32[:], int32[:], int32[:]),
        (int64, int64, int64, int32, int16[:, :], int64[:], int64[:], int64[:]),
    ],
    "(),(),(),(),(n,m)->(n),(n),(n)",
)
def _neighbours_numba(x, y, f, nside, offsets, nx, ny, nf):
    for index, (x_offset, y_offset) in enumerate(offsets):
        cx, cy, cf = adjust_xyf(x + x_offset, y + y_offset, f, nside)

        nx[index] = cx
        ny[index] = cy
        nf[index] = cf


def _neighbours(cell_ids, *, offsets, nside, indexing_scheme):
    x, y, face = cell_ids2xyf(cell_ids, nside=nside, indexing_scheme=indexing_scheme)
    neighbour_x, neighbour_y, neighbour_face = _neighbours_numba(
        x, y, face, nside, offsets
    )

    n_ = xyf2cell_ids(
        neighbour_x,
        neighbour_y,
        neighbour_face,
        nside=nside,
        indexing_scheme=indexing_scheme,
    )
    return np.where(neighbour_face == -1, -1, n_)


def minimum_dtype(value):
    if value < np.iinfo("int8").max:
        return "int8"
    elif value < np.iinfo("int16").max:
        return "int16"
    elif value < np.iinfo("int32").max:
        return "int32"
    elif value < np.iinfo("int64").max:
        return "int64"


def neighbours(cell_ids, *, resolution, indexing_scheme, ring=1):
    """determine the neighbours within the nth ring around the center pixel

    Parameters
    ----------
    resolution : int
        The healpix resolution. Has to be within [0, 29].
    cell_ids : array-like
        The cell ids of which to find the neighbours.
    ring : int, default: 1
        The number of the ring. `ring=0` returns just the cell id, `ring=1` returns the 8
        (or 7) immediate neighbours, `ring=2` returns the 8 (or 7) immediate neighbours
        plus their immediate neighbours (a total of 24 cells), and so on.
    """
    nside = 2**resolution
    if ring < 0:
        raise ValueError(f"ring must be a positive integer or 0, got {ring}")
    if ring > nside:
        raise ValueError(
            "rings containing more than the neighbouring base pixels are not supported"
        )

    offsets = np.asarray(list(generate_offsets(ring=ring)), dtype=minimum_dtype(ring))

    if isinstance(cell_ids, dask_array_type):
        n_neighbours = (2 * ring + 1) ** 2
        return da.map_blocks(
            _neighbours,
            cell_ids,
            offsets=offsets,
            nside=nside,
            indexing_scheme=indexing_scheme,
            new_axis=1,
            chunks=cell_ids.chunks + (n_neighbours,),
            dtype=cell_ids.dtype,
        )
    else:
        return _neighbours(
            cell_ids, offsets=offsets, nside=nside, indexing_scheme=indexing_scheme
        )
