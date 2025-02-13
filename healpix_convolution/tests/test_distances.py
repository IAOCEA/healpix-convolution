import numpy as np
import pytest
import xdggs

import healpix_convolution.distances as hds


def test_angle_between_vectors():
    angles = np.linspace(0, 2 * np.pi, 10)
    vectors = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1)

    actual = hds.angle_between_vectors(vectors[:1, :], vectors, axis=-1)
    expected = np.pi - abs(angles - np.pi)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ["neighbours", "expected"],
    (
        (np.array([[1, 0, 2, 3]]), np.array([[0, 0.25637566, 0.3699723, 0.25574741]])),
        (np.array([[2, 71, 8]]), np.array([[0, 0.25637566, 0.25574741]])),
        (np.array([[2, 71], [2, 8]]), np.array([[0, 0.25637566], [0, 0.25574741]])),
    ),
)
def test_angular_distances_numpy(neighbours, expected):
    grid_info = xdggs.HealpixInfo(level=2, indexing_scheme="nested")
    actual = hds.angular_distances(neighbours, grid_info=grid_info)

    np.testing.assert_allclose(actual, expected)
