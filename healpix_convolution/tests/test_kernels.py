import numpy as np
import pytest

from healpix_convolution import kernels as np_kernels


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
    input_cell_ids = np.unique(neighbours)
    if input_cell_ids[0] == -1:
        input_cell_ids = input_cell_ids[1:]

    actual = np_kernels.common.create_sparse(cell_ids, neighbours, weights)

    nnz = np.sum(neighbours != -1, axis=1)
    value = nnz * weights[0]

    expected_shape = (cell_ids.size, input_cell_ids.size)
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
        actual = np_kernels.gaussian_kernel(cell_ids, **kwargs)

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
