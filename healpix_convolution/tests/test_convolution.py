import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import sparse
from hypothesis import given, settings

from healpix_convolution import convolution


@given(
    data=npst.arrays(
        shape=st.sampled_from([(5,), (10, 5)]),
        # TODO: figure out how to deal with floating point values
        dtype=st.sampled_from(["int32", "int64", "float32", "float64"]),
        elements={
            "allow_infinity": False,
            "allow_subnormal": False,
            "min_value": 1e9,
            "max_value": 1e9,
        },
    ),
)
@settings(deadline=1000)
def test_numpy_convolve(data):
    dense_kernel = (
        np.array(
            [
                [1, 1, 0, 0, 1],
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [1, 0, 0, 1, 1],
            ]
        )
        / 3
    )

    kernel = sparse.COO.from_numpy(dense_kernel, fill_value=0)
    actual = convolution.convolve(data, kernel)

    padding = [(0, 0)] * (data.ndim - 1) + [(1, 1)]
    padded = np.pad(data, padding, mode="wrap")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 3, axis=-1)
    expected = np.mean(windows, axis=-1)

    np.testing.assert_allclose(actual, expected)


def test_xarray_convolve_unrelated_vars():
    import xarray as xr

    from healpix_convolution.xarray import convolution as xr_convolution

    dense_kernel = (
        np.array(
            [
                [1, 1, 0, 0, 1],
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [1, 0, 0, 1, 1],
            ]
        )
        / 3
    )

    kernel_arr = sparse.COO.from_numpy(dense_kernel, fill_value=0)
    grid_info = {"grid_name": "healpix", "level": 5, "indexing_scheme": "nested"}
    cell_ids = np.array([0, 1, 2, 3, 4], dtype="uint64")

    kernel = xr.DataArray(
        kernel_arr,
        dims=["output_cells", "input_cells"],
        coords={
            "input_cell_ids": ("input_cells", cell_ids, grid_info),
            "output_cell_ids": ("output_cells", cell_ids, grid_info),
        },
    )

    data = np.ones((20, kernel.shape[1]), dtype=float)
    ds = xr.Dataset(
        {"data": (["time", "cells"], data)},
        coords={"cell_ids": ("cells", cell_ids), "time": np.arange(20, 40)},
    ).dggs.decode(grid_info)

    actual = xr_convolution.convolve(ds, kernel, mode="mean")

    assert set(actual.variables) == set(ds.variables)
