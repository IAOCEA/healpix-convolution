import numpy as np
import pytest

from healpix_convolution import kernels


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
    actual = kernels.common.create_sparse(cell_ids, neighbours, weights, shape=(48, 48))

    nnz = np.sum(neighbours != -1, axis=1)
    value = nnz * weights[0]

    assert hasattr(actual, "nnz"), "not a sparse matrix"
    assert np.allclose(
        np.sum(actual[cell_ids, :], axis=1).todense(), value
    ), "rows have unexpected values"
    assert actual.size == 48**2
