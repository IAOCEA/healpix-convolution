from dataclasses import dataclass

import numpy as np
from xarray.namedarray._typing import _arrayfunction_or_api as _ArrayLike
from xarray.namedarray._typing import _ScalarType
from xdggs.grid import DGGSInfo


@dataclass
class Padding:
    cell_ids: _ArrayLike
    insert_indices: _ArrayLike
    grid_info: DGGSInfo

    def apply(self, data):
        raise NotImplementedError()


@dataclass
class ConstantPadding(Padding):
    constant_value: _ScalarType

    def apply(self, data):
        return np.insert(data, self.insert_indices, self.constant_value)


@dataclass
class LinearRampPadding(Padding):
    end_value: _ScalarType
    border_indices: _ArrayLike
    distance: _ArrayLike

    def apply(self, data):
        offsets = data[..., self.border_indices]
        ramp = self.end_value - offsets
        pad_values = offsets + ramp * self.distance

        return np.insert(data, self.insert_indices, pad_values, axis=-1)


@dataclass
class DataPadding(Padding):
    data_indices: _ArrayLike

    def apply(self, data):
        pad_values = data[..., self.data_indices]

        return np.insert(data, self.insert_indices, pad_values, axis=-1)


def pad(cell_ids, *, grid_info, ring, mode="constant", constant_value=0, end_value=0):
    """pad an array

    Parameters
    ----------
    cell_ids : array-like
        The cell ids.
    data : array-like
        The array to pad.
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
    constant_value : scalar, default: 0
        The constant value used in constant mode.
    end_value : scalar, default: 0
        The othermost value to interpolate to.

    Returns
    -------
    padding_object : Padding
        The padding object. Can be used to apply the same padding operation for different
        arrays with the same geometry.
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
