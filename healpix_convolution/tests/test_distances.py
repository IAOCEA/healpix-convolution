import numpy as np
import pytest

import healpix_convolution.distances as hds


@pytest.mark.parametrize(
    ["neighbours", "expected"],
    (
        (np.array([[1, 0, 2, 3]]), np.array([[0, 0.25637566, 0.3699723, 0.25574741]])),
        (np.array([[2, 71, 8]]), np.array([[0, 0.25637566, 0.25574741]])),
    ),
)
def test_distances_numpy(neighbours, expected):
    actual = hds.angular_distances(neighbours, resolution=2, indexing_scheme="nested")

    np.testing.assert_allclose(actual, expected)
