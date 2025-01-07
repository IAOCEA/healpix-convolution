from dataclasses import dataclass, field
from functools import partial

import numpy as np
from xarray.namedarray._typing import _arrayfunction_or_api as _ArrayLike
from xarray.namedarray._typing import _ScalarType
from xdggs.grid import DGGSInfo

from healpix_convolution.neighbours import neighbours as search_neighbours


@dataclass
class Padding:
    """Base class for all cached padding operations

    Parameters
    ----------
    cell_ids: array_like
        The cell ids of the padding cells
    insert_indices: array_like
        Indices of where to insert the padding cells to keep the input cell ids sorted.
    grid_info: xdggs.DGGSInfo
        The grid parameters
    """

    cell_ids: _ArrayLike = field(repr=False)
    insert_indices: _ArrayLike = field(repr=False)
    grid_info: DGGSInfo = field(repr=False)

    def apply(self, data):
        """apply the padding to the data"""
        raise NotImplementedError()


@dataclass
class ConstantPadding(Padding):
    constant_value: _ScalarType

    def apply(self, data):
        common_dtype = np.result_type(data, self.constant_value)

        if hasattr(data, "chunks"):
            import dask.array as da

            values = da.full_like(
                data,
                shape=self.insert_indices.shape,
                dtype=common_dtype,
                fill_value=self.constant_value,
                chunks=data.chunks[0][-1],
            )
        else:
            values = self.constant_value

        return np.insert(
            data.astype(common_dtype), self.insert_indices, values, axis=-1
        )


@dataclass
class LinearRampPadding(Padding):
    end_value: _ScalarType
    ring: int

    border_indices: _ArrayLike
    distance: _ArrayLike

    def apply(self, data):
        offsets = data[..., self.border_indices]
        ramp = (self.end_value - offsets) / self.ring
        pad_values = offsets + ramp * self.distance

        return np.insert(data, self.insert_indices, pad_values, axis=-1)


@dataclass
class AggregationPadding(Padding):
    """
    Pad by aggregating neighbouring pixels

    Parameters
    ----------
    agg : callable
        The aggregation function. Used to aggregate over the second dimension
    data_indices : array-like
        The data indices to aggregate over. Must have 2 dimensions, with the first having
        the same size as ``insert_indices``.
    """

    agg: callable
    data_indices: _ArrayLike

    def apply(self, data):
        mask = self.data_indices != -1

        new_shape = data.shape[:-1] + self.data_indices.shape
        data_values_ = np.reshape(
            data[..., np.reshape(self.data_indices, (-1,))], new_shape
        )
        data_values = np.where(mask, data_values_, np.nan)

        pad_values = self.agg(data_values, axis=-1)

        return np.insert(data, self.insert_indices, pad_values, axis=-1)


@dataclass
class DataPadding(Padding):
    data_indices: _ArrayLike

    def apply(self, data):
        pad_values = data[..., self.data_indices]

        return np.insert(data, self.insert_indices, pad_values, axis=-1)


def constant_mode(cell_ids, neighbours, grid_info, constant_value):
    all_cell_ids = np.unique(neighbours)
    if all_cell_ids[0] == -1:
        all_cell_ids = all_cell_ids[1:]

    new_cell_ids = np.setdiff1d(
        all_cell_ids, np.concatenate((np.array([-1]), cell_ids))
    )

    insert_indices = np.searchsorted(cell_ids, new_cell_ids)

    return ConstantPadding(
        cell_ids=all_cell_ids,
        insert_indices=insert_indices,
        grid_info=grid_info,
        constant_value=constant_value,
    )


def linear_ramp_mode(cell_ids, neighbours, grid_info, end_value):
    # algorithm:  for each padded cell find the closest edge cell and the distance
    pass


def edge_mode(cell_ids, neighbours, grid_info):
    # algorithm: for each padded cell find the closest edge cell
    pass


def reflect_mode(cell_ids, neighbours, grid_info):
    # algorithm: for each padded cell, find the closest edge cell and the distance, then take the index of the cell that in the same distance and direction as the edge cell from the padded cell
    pass


def agg_mode(cell_ids, neighbours, grid_info, *, agg, ring):
    all_cell_ids = np.unique(neighbours)
    new_cell_ids = np.setdiff1d(
        all_cell_ids, np.concatenate((np.array([-1]), cell_ids))
    )

    pad_neighbours = search_neighbours(
        new_cell_ids,
        resolution=grid_info.level,
        indexing_scheme=grid_info.indexing_scheme,
        ring=ring,
    )
    mask = np.isin(pad_neighbours, cell_ids)

    data_indices = np.where(mask, np.searchsorted(cell_ids, pad_neighbours), -1)
    insert_indices = np.searchsorted(cell_ids, new_cell_ids)

    return AggregationPadding(
        cell_ids=all_cell_ids,
        insert_indices=insert_indices,
        grid_info=grid_info,
        agg=agg,
        data_indices=data_indices,
    )


def pad(
    cell_ids,
    *,
    grid_info,
    ring,
    mode="constant",
    constant_value=0,
    end_value=0,
    reflect_type="even",
):
    """pad an array

    Parameters
    ----------
    cell_ids : array-like
        The cell ids.
    grid_info : xdggs.DGGSInfo
        The grid parameters.
    ring : int
        The pad width in rings around the input domain. Must be 0 or positive.
    mode : str, default: "constant"
        The padding mode. Can be one of:

        - "constant": fill the padded cells with ``constant_value``.
        - "linear_ramp": linearly interpolate the padded cells from the edge of the array
          to ``end_value``. For ring 1, this is the same as ``mode="constant"``
        - "edge": fill the padded cells with the values at the edge of the array.
        - "reflect": pad with the reflected values.
        - "mean": pad with the mean of the neighbours.
        - "minimum": pad with the minimum of the neighbours.
        - "maximum": pad with the maximum of the neighbours.
    constant_value : scalar, default: 0
        The constant value used in constant mode.
    end_value : scalar, default: 0
        The othermost value to interpolate to. Only used in linear ramp mode.
    reflect_type : {"even", "odd"}, default: "even"
        The reflect type. Only used in reflect mode.
    initial : scalar, default: 0
        The identity to use in maximum / minimum mode.

    Returns
    -------
    padding_object : healpix_convolution.Padding
        The padding object. Can be used to apply the same padding operation for different
        arrays with the same geometry.

    Warnings
    --------
    This assumes ``cell_ids`` is sorted, and will give wrong results if it is not.
    """
    # TODO: figure out how to allow reusing indices. How this works depends on the mode:
    # - in constant mode, we have:
    #   * an array of new cell ids
    #   * an array of indices that indicate where to insert them
    #   * and the constant value
    # - in the case of linear ramp, we have:
    #   * the new cell ids
    #   * an array of indices that indicate where to insert them
    #   * the value of where the ramp should end
    #   * the distance of each cell from the edge of the array
    # - in all other cases, we have:
    #   * an array of new cell ids
    #   * an array of indices that indicate where to insert them
    #   * an array of indices that map existing values to the new cell ids
    # To be able to reuse this, we need a set of dataclasses that can encapsulate that,
    # plus a method to apply the padding to data.
    neighbours = search_neighbours(
        cell_ids,
        resolution=grid_info.level,
        indexing_scheme=grid_info.indexing_scheme,
        ring=ring,
    )

    modes = {
        "constant": partial(constant_mode, constant_value=constant_value),
        "linear_ramp": partial(linear_ramp_mode, end_value=end_value, ring=ring),
        "edge": edge_mode,
        "reflect_mode": partial(reflect_mode, reflect_type=reflect_type),
        "mean": partial(agg_mode, agg=np.nanmean, ring=ring),
        "maximum": partial(agg_mode, agg=np.nanmax, ring=ring),
        "minimum": partial(agg_mode, agg=np.nanmin, ring=ring),
        "median": partial(agg_mode, agg=np.nanmedian, ring=ring),
    }

    mode_func = modes.get(mode)
    if mode_func is None:
        raise ValueError(f"unknown mode: {mode}")

    return mode_func(cell_ids, neighbours, grid_info)
