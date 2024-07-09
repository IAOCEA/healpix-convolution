import numpy as np
import pytest
import xarray as xr
import xdggs

from healpix_convolution import kernels as np_kernels
from healpix_convolution.xarray import kernels as xr_kernels


@pytest.mark.parametrize(
    ["cell_ids", "neighbours", "weights"],
    (
        pytest.param(
            np.array([0, 1]),
            np.array(
                [[0, 17, 19, 2, 3, 1, 23, 22, 35], [3, 2, 13, 15, 11, 7, 6, 1, 0]]
            ),
            np.full((18,), fill_value=1 / 18),
        ),
        pytest.param(
            np.array([3, 2]),
            np.array(
                [[3, 2, 13, 15, 11, 7, 6, 1, 0], [2, 19, -1, 13, 15, 3, 1, 0, 17]]
            ),
            np.full((18,), fill_value=1 / 17),
        ),
    ),
)
def test_create_sparse(cell_ids, neighbours, weights):
    expected_cell_ids = np.unique(neighbours)
    if expected_cell_ids[0] == -1:
        expected_cell_ids = expected_cell_ids[1:]

    actual_cell_ids, actual = np_kernels.common.create_sparse(
        cell_ids, neighbours, weights
    )

    nnz = np.sum(neighbours != -1, axis=1)
    value = nnz * weights[0]

    np.testing.assert_equal(actual_cell_ids, expected_cell_ids)

    expected_shape = (cell_ids.size, expected_cell_ids.size)
    assert hasattr(actual, "nnz"), "not a sparse matrix"
    assert np.allclose(
        np.sum(actual, axis=1).todense(), value
    ), "rows have unexpected values"
    assert actual.shape == expected_shape


class TestGaussian:
    @pytest.mark.parametrize(
        ["cell_ids", "kwargs"],
        (
            (
                np.array([1, 2]),
                {"resolution": 1, "indexing_scheme": "nested", "sigma": 0.1},
            ),
            (
                np.array([1, 2]),
                {"resolution": 1, "indexing_scheme": "ring", "sigma": 0.1},
            ),
            (
                np.array([0, 2]),
                {"resolution": 1, "indexing_scheme": "nested", "sigma": 0.2},
            ),
            (
                np.array([1, 2]),
                {
                    "resolution": 1,
                    "indexing_scheme": "nested",
                    "sigma": 0.1,
                    "kernel_size": 5,
                },
            ),
            (
                np.array([3, 0]),
                {
                    "resolution": 1,
                    "indexing_scheme": "ring",
                    "sigma": 0.1,
                    "kernel_size": 3,
                },
            ),
        ),
    )
    def test_gaussian_kernel(self, cell_ids, kwargs):
        _, actual = np_kernels.gaussian_kernel(cell_ids, **kwargs)

        kernel_sum = np.sum(actual, axis=1)

        assert np.sum(np.isnan(actual)) == 0
        np.testing.assert_allclose(kernel_sum.todense(), 1)

        # try determining the sigma from the values for better tests

    @pytest.mark.parametrize(
        ["cell_ids", "kwargs", "error", "pattern"],
        (
            (
                np.array([[0, 1], [2, 3]]),
                {
                    "resolution": 1,
                    "indexing_scheme": "nested",
                    "sigma": 0.1,
                    "kernel_size": 3,
                },
                ValueError,
                "1-dimensional",
            ),
            (
                np.array([0, 1]),
                {
                    "resolution": 1,
                    "indexing_scheme": "nested",
                    "sigma": 0.1,
                    "kernel_size": 7,
                },
                ValueError,
                "more than the neighbouring base pixels",
            ),
        ),
    )
    def test_gaussian_kernel_errors(self, cell_ids, kwargs, error, pattern):
        with pytest.raises(error, match=pattern):
            np_kernels.gaussian_kernel(cell_ids, **kwargs)


class TestXarray:
    @pytest.mark.parametrize(
        ["obj", "kwargs"],
        (
            (
                xr.DataArray(
                    [1, 2],
                    coords={
                        "cell_ids": (
                            "cells",
                            np.array([1, 2]),
                            {
                                "grid_name": "healpix",
                                "resolution": 1,
                                "indexing_scheme": "nested",
                            },
                        )
                    },
                    dims="cells",
                ),
                {"sigma": 0.1},
            ),
            (
                xr.DataArray(
                    [1, 2],
                    coords={
                        "cell_ids": (
                            "cells",
                            np.array([1, 2]),
                            {
                                "grid_name": "healpix",
                                "resolution": 1,
                                "indexing_scheme": "ring",
                            },
                        )
                    },
                    dims="cells",
                ),
                {"sigma": 0.1},
            ),
            (
                xr.DataArray(
                    [0, 2],
                    coords={
                        "cell_ids": (
                            "cells",
                            np.array([0, 2]),
                            {
                                "grid_name": "healpix",
                                "resolution": 1,
                                "indexing_scheme": "nested",
                            },
                        )
                    },
                    dims="cells",
                ),
                {"sigma": 0.2},
            ),
            (
                xr.DataArray(
                    [1, 2],
                    coords={
                        "cell_ids": (
                            "cells",
                            np.array([1, 2]),
                            {
                                "grid_name": "healpix",
                                "resolution": 1,
                                "indexing_scheme": "nested",
                            },
                        )
                    },
                    dims="cells",
                ),
                {"sigma": 0.1, "kernel_size": 5},
            ),
            (
                xr.DataArray(
                    [0, 3],
                    coords={
                        "cell_ids": (
                            "cells",
                            np.array([0, 3]),
                            {
                                "grid_name": "healpix",
                                "resolution": 1,
                                "indexing_scheme": "ring",
                            },
                        )
                    },
                    dims="cells",
                ),
                {"sigma": 0.1, "kernel_size": 3},
            ),
        ),
    )
    def test_gaussian_kernel(self, obj, kwargs):
        obj_ = obj.pipe(xdggs.decode)
        actual = xr_kernels.gaussian_kernel(obj_, **kwargs)

        kernel_sum = actual.sum(dim="input_cells")

        assert actual.count() == actual.size
        np.testing.assert_allclose(kernel_sum.data.todense(), 1)
