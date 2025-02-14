from functools import partial

import healpix_geo

try:
    import dask.array as da

    dask_array_type = (da.Array,)
except ImportError:  # pragma: no cover
    dask_array_type = ()
    da = None


def neighbours(cell_ids, *, grid_info, ring=1):
    """determine the neighbours within the nth ring around the center pixel

    Parameters
    ----------
    cell_ids : array-like
        The cell ids of which to find the neighbours.
    grid_info : xdggs.HealpixInfo
        The grid parameters.
    ring : int, default: 1
        The number of the ring. `ring=0` returns just the cell id, `ring=1` returns the 8
        (or 7) immediate neighbours, `ring=2` returns the 8 (or 7) immediate neighbours
        plus their immediate neighbours (a total of 24 cells), and so on.
    """
    nside = grid_info.nside
    if ring < 0:
        raise ValueError(f"ring must be a positive integer or 0, got {ring}")
    if ring > nside:
        raise ValueError(
            "rings containing more than the neighbouring base pixels are not supported"
        )

    if grid_info.indexing_scheme == "nested":
        neighbours_disk = healpix_geo.nested.neighbours_disk
    elif grid_info.indexing_scheme == "ring":
        neighbours_disk = healpix_geo.ring.neighbours_disk
    else:
        raise ValueError(f"unsupported indexing scheme: '{grid_info.indexing_scheme}'")

    f = partial(neighbours_disk, depth=grid_info.level, ring=ring)
    if isinstance(cell_ids, dask_array_type):
        n_neighbours = (2 * ring + 1) ** 2
        return da.map_blocks(
            f,
            cell_ids,
            new_axis=1,
            chunks=cell_ids.chunks + (n_neighbours,),
            dtype=cell_ids.dtype,
        )
    else:
        return f(cell_ids)
