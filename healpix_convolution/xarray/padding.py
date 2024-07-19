from dataclasses import dataclass

import xarray as xr
import xdggs  # noqa: F401

from healpix_convolution import padding, utils


@dataclass
class Padding:
    padding: padding.Padding

    def _apply(self, ds):
        to_drop = [name for name, coord in ds.coords.items() if "cells" in coord.dims]
        cell_ids = xr.Variable(
            "cells", self.padding.cell_ids, attrs=ds["cell_ids"].attrs
        )

        return (
            ds.drop_vars(to_drop)
            .map(self.padding.apply)
            .assign_coords(cell_ids=cell_ids)
        )

    def apply(self, obj):
        return utils.call_on_dataset(self._apply, obj)


def pad(
    cell_ids,
    *,
    ring,
    mode="constant",
    constant_value=0,
    end_value=0,
    reflect_type="even",
):
    """pad a xarray object

    Parameters
    ----------
    cell_ids : xarray.DataArray
        The cell ids to pad. Must have a `xdggs` index.
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
        The othermost value to interpolate to. Only used in linear ramp mode.
    reflect_type : {"even", "odd"}, default: "even"
        The reflect type. Only used in reflect mode.

    Returns
    -------
    padded : Padding
        The padding object. Can be used to apply the same padding operation for multiple
        objects with the same geometry.
    """
    raw_padding = padding.pad(
        cell_ids.data,
        grid_info=cell_ids.dggs.grid_info,
        ring=ring,
        mode=mode,
        constant_value=constant_value,
        end_value=end_value,
        reflect_type=reflect_type,
    )
    return Padding(raw_padding)
