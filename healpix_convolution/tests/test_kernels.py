import numpy as np
import pytest
import xarray as xr
import xdggs

from healpix_convolution import kernels as np_kernels
from healpix_convolution.kernels.gaussian import compute_ring, healpix_resolution
from healpix_convolution.xarray import kernels as xr_kernels


@pytest.mark.parametrize(
    "level, expected",
    [
        # level = 0 --> sqrt(4*pi/(12)) --> sqrt(pi/3)
        (0, np.sqrt(np.pi / 3)),
        # level = 1 --> sqrt(4*pi/(12*4)) --> sqrt(pi/12)
        (1, np.sqrt(np.pi / 12)),
    ],
)
def test_healpix_resolution(level, expected):
    result = healpix_resolution(level)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "level, sigma, truncate, kernel_size, expected",
    [
        # kernel_size = 5 --> ring = 5 // 2 = 2.
        (2, 1.0, 1.0, 5, 2),
        (2, 0.5, 2.0, 6, 3),
        # kernel_size = None
        #   level = 1  --> cell_distance = healpix_resolution(1) = sqrt(pi/12) = 0.51168
        #   sigma = 1.0, truncate = 2.0 --> (2*1.0/0.51168) = 3.91 (3.91 // 2) = 1
        (1, 1.0, 2.0, None, 1),
        #   level = 0  => cell_distance = healpix_resolution(0) = sqrt(pi/3) = 1.02333,
        #   sigma = 2.0, truncate = 3.0 => (3*2/1.02333) = 5.86 (5.86 // 2) = 2
        (0, 2.0, 3.0, None, 2),
        # verify ring equal 1 even if small value:
        (5, 0.01, 1.0, None, 1),
    ],
)
def test_compute_ring(level, sigma, truncate, kernel_size, expected):
    result = compute_ring(level, sigma, truncate, kernel_size)
    assert result == expected


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


@pytest.mark.parametrize(
    ["cell_ids", "neighbours", "threshold"],
    (
        pytest.param(
            np.array([0, 1]),
            np.array(
                [[0, 17, 19, 2, 3, 1, 23, 22, 35], [3, 2, 13, 15, 11, 7, 6, 1, 0]]
            ),
            None,
        ),
        pytest.param(
            np.array([0, 1]),
            np.array(
                [[0, 17, 19, 2, 3, 1, 23, 22, 35], [3, 2, 13, 15, 11, 7, 6, 1, 0]]
            ),
            0.1,
        ),
        pytest.param(
            np.array([3, 2]),
            np.array(
                [[3, 2, 13, 15, 11, 7, 6, 1, 0], [2, 19, -1, 13, 15, 3, 1, 0, 17]]
            ),
            0.1,
        ),
    ),
)
def test_create_sparse_threshold(cell_ids, neighbours, threshold):
    expected_cell_ids = np.unique(neighbours)
    if expected_cell_ids[0] == -1:
        expected_cell_ids = expected_cell_ids[1:]

    weights = np.reshape(neighbours / np.sum(neighbours, axis=1, keepdims=True), (-1,))

    actual_cell_ids, actual = np_kernels.common.create_sparse(
        cell_ids, neighbours, weights, weights_threshold=threshold
    )

    expected_nnz = (
        np.sum(abs(weights) >= threshold) if threshold is not None else weights.size
    )

    np.testing.assert_equal(actual_cell_ids, expected_cell_ids)

    expected_shape = (cell_ids.size, expected_cell_ids.size)
    assert hasattr(actual, "nnz"), "not a sparse matrix"
    assert actual.shape == expected_shape
    assert actual.nnz == expected_nnz, "non-zero entries don't match"


def fit_polynomial(x, y, deg):
    mask = y > 0
    x_ = x[mask]
    y_ = y[mask]

    p = np.polynomial.Polynomial.fit(x_, np.log(y_), deg=deg)
    return p.convert()


def reconstruct_sigma(
    cell_ids,
    kernel,
    *,
    grid_info,
    sigma,
    truncate=4.0,
    kernel_size=None,
):
    from healpix_convolution import angular_distances, neighbours

    ring = np_kernels.gaussian.compute_ring(
        grid_info.level, sigma, truncate, kernel_size
    )

    nb = neighbours(cell_ids, grid_info=grid_info, ring=ring)
    distances = angular_distances(nb, grid_info=grid_info)

    _, distances_ = np_kernels.common.create_sparse(cell_ids, nb, distances)

    x = distances_.todense()
    y = kernel.todense()

    polynomials = [
        fit_polynomial(x[n, :], y[n, :], deg=2) for n in range(cell_ids.size)
    ]
    return np.array([np.sqrt(-1 / 2 / p.coef[2]) for p in polynomials])


class TestGaussian:
    @pytest.mark.parametrize(
        ["cell_ids", "kwargs"],
        (
            (
                np.array([1, 2]),
                {"level": 1, "indexing_scheme": "nested", "sigma": 0.1},
            ),
            (
                np.array([1, 2]),
                {"level": 1, "indexing_scheme": "ring", "sigma": 0.1},
            ),
            (
                np.array([0, 2]),
                {"level": 1, "indexing_scheme": "nested", "sigma": 0.2},
            ),
            (
                np.array([1, 2]),
                {
                    "level": 1,
                    "indexing_scheme": "nested",
                    "sigma": 0.1,
                    "kernel_size": 5,
                },
            ),
            (
                np.array([3, 0]),
                {
                    "level": 1,
                    "indexing_scheme": "ring",
                    "sigma": 0.1,
                    "kernel_size": 3,
                },
            ),
        ),
    )
    def test_gaussian_kernel(self, cell_ids, kwargs):
        grid_info = xdggs.HealpixInfo(
            level=kwargs.pop("level"), indexing_scheme=kwargs.pop("indexing_scheme")
        )
        _, actual = np_kernels.gaussian_kernel(cell_ids, grid_info=grid_info, **kwargs)

        kernel_sum = np.sum(actual, axis=1)

        assert np.sum(np.isnan(actual)) == 0
        np.testing.assert_allclose(kernel_sum.todense(), 1)

        # try determining the sigma from the values for better tests
        reconstructed_sigma = reconstruct_sigma(
            cell_ids, actual, grid_info=grid_info, **kwargs
        )
        np.testing.assert_allclose(reconstructed_sigma, kwargs["sigma"])

    @pytest.mark.parametrize(
        ["cell_ids", "kwargs", "error", "pattern"],
        (
            (
                np.array([[0, 1], [2, 3]]),
                {
                    "level": 1,
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
                    "level": 1,
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
        grid_info = xdggs.HealpixInfo(
            level=kwargs.pop("level"), indexing_scheme=kwargs.pop("indexing_scheme")
        )
        with pytest.raises(error, match=pattern):
            np_kernels.gaussian_kernel(cell_ids, grid_info=grid_info, **kwargs)


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
                                "level": 1,
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
                                "level": 1,
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
                                "level": 1,
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
                                "level": 1,
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
                                "level": 1,
                                "indexing_scheme": "ring",
                            },
                        )
                    },
                    dims="cells",
                ),
                {"sigma": 0.1, "kernel_size": 3},
            ),
            (
                xr.DataArray(
                    np.arange(12 * 4**2, dtype="int64"),
                    coords={
                        "cell_ids": (
                            "cells",
                            np.arange(12 * 4**2, dtype="int64"),
                            {
                                "grid_name": "healpix",
                                "level": 2,
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

        grid_info = obj_.dggs.grid_info
        reconstructed_sigma = reconstruct_sigma(
            obj.cell_ids.data,
            actual.data,
            grid_info=grid_info,
            **kwargs,
        )
        np.testing.assert_allclose(reconstructed_sigma, kwargs["sigma"])
