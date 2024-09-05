import numpy as np
import pytest
import xarray as xr
import xdggs

from healpix_convolution import padding
from healpix_convolution.xarray import padding as xr_padding

try:
    import dask.array as da
except ImportError:  # pragma: no cover
    da = None
    dask_array_type = ()

    dask_available = False
else:
    dask_array_type = da.Array
    dask_available = True

requires_dask = pytest.mark.skipif(not dask_available, reason="requires dask")


class TestArray:
    @pytest.mark.parametrize(
        ["ring", "mode", "kwargs", "expected_cell_ids", "expected_data"],
        (
            pytest.param(
                1,
                "constant",
                {"constant_value": np.nan},
                np.array([163, 166, 167, 169, 171, 172, 173, 174, 175, 178, 184, 186]),
                np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        1,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
                id="constant-ring1-nan",
            ),
            pytest.param(
                2,
                "constant",
                {"constant_value": 0},
                np.array(
                    [
                        160,
                        161,
                        162,
                        163,
                        164,
                        165,
                        166,
                        167,
                        168,
                        169,
                        170,
                        171,
                        172,
                        173,
                        174,
                        175,
                        176,
                        177,
                        178,
                        179,
                        184,
                        185,
                        186,
                        187,
                        853,
                        855,
                        861,
                        863,
                        885,
                        887,
                    ]
                ),
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
                id="constant-ring2-0",
            ),
            pytest.param(
                1,
                "mean",
                {},
                np.array([163, 166, 167, 169, 171, 172, 173, 174, 175, 178, 184, 186]),
                np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                id="mean-ring1",
            ),
            pytest.param(
                1,
                "minimum",
                {},
                np.array([163, 166, 167, 169, 171, 172, 173, 174, 175, 178, 184, 186]),
                np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                id="minimum-ring1",
            ),
        ),
    )
    @pytest.mark.parametrize(
        "dask",
        (
            False,
            pytest.param(True, marks=requires_dask),
        ),
    )
    def test_pad(self, dask, ring, mode, kwargs, expected_cell_ids, expected_data):
        grid_info = xdggs.healpix.HealpixInfo(resolution=4, indexing_scheme="nested")
        cell_ids = np.array([172, 173])

        if not dask:
            data = np.full_like(cell_ids, fill_value=1)
            expected = expected_data
        else:
            import dask.array as da

            data = da.full_like(cell_ids, fill_value=1, chunks=(1,))
            expected = da.from_array(expected_data, chunks=(1,))

        padder = padding.pad(
            cell_ids, grid_info=grid_info, ring=ring, mode=mode, **kwargs
        )
        actual = padder.apply(data)

        if dask:
            assert isinstance(actual, da.Array)
            assert actual.chunks == expected.chunks, "chunksizes differ"

        np.testing.assert_equal(padder.cell_ids, expected_cell_ids)
        np.testing.assert_equal(actual, expected_data)


class TestXarray:
    @pytest.mark.parametrize(
        ["ring", "mode", "kwargs", "expected_cell_ids", "expected_data"],
        (
            pytest.param(
                1,
                "constant",
                {"constant_value": np.nan},
                np.array([163, 166, 167, 169, 171, 172, 173, 174, 175, 178, 184, 186]),
                np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        1,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
                id="constant-ring1-nan",
            ),
            pytest.param(
                2,
                "constant",
                {"constant_value": 0},
                np.array(
                    [
                        160,
                        161,
                        162,
                        163,
                        164,
                        165,
                        166,
                        167,
                        168,
                        169,
                        170,
                        171,
                        172,
                        173,
                        174,
                        175,
                        176,
                        177,
                        178,
                        179,
                        184,
                        185,
                        186,
                        187,
                        853,
                        855,
                        861,
                        863,
                        885,
                        887,
                    ]
                ),
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
                id="constant-ring2-0",
            ),
            pytest.param(
                1,
                "mean",
                {},
                np.array([163, 166, 167, 169, 171, 172, 173, 174, 175, 178, 184, 186]),
                np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                id="mean-ring1",
            ),
            pytest.param(
                1,
                "minimum",
                {},
                np.array([163, 166, 167, 169, 171, 172, 173, 174, 175, 178, 184, 186]),
                np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                id="minimum-ring1",
            ),
        ),
    )
    @pytest.mark.parametrize(
        "dask",
        (
            False,
            pytest.param(True, marks=requires_dask),
        ),
    )
    @pytest.mark.parametrize("type_", (xr.Dataset, xr.DataArray))
    def test_pad(
        self, dask, ring, mode, kwargs, expected_cell_ids, expected_data, type_
    ):
        grid_info = xdggs.healpix.HealpixInfo(resolution=4, indexing_scheme="nested")
        cell_ids = np.array([172, 173])

        if not dask:
            data_ = np.full_like(cell_ids, fill_value=1)
            expected_data_ = expected_data
        else:
            import dask.array as da

            data_ = da.full_like(cell_ids, fill_value=1, chunks=(1,))
            expected_data_ = da.from_array(expected_data, chunks=(1,))

        ds = xr.Dataset(
            {"data": ("cells", data_)},
            coords={"cell_ids": ("cells", cell_ids, grid_info.to_dict())},
        ).pipe(xdggs.decode)
        expected_ds = xr.Dataset(
            {"data": ("cells", expected_data_)},
            coords={"cell_ids": ("cells", expected_cell_ids, grid_info.to_dict())},
        ).pipe(xdggs.decode)

        if type_ is xr.Dataset:
            data = ds
            expected = expected_ds
        else:
            data = ds["data"]
            expected = expected_ds["data"]

        padder = xr_padding.pad(data["cell_ids"], ring=ring, mode=mode, **kwargs)
        actual = padder.apply(data)

        if dask:
            assert actual.chunksizes == expected.chunksizes, "chunksizes differ"

        xr.testing.assert_identical(actual, expected)
