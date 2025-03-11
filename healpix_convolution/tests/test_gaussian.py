import pytest
import numpy as np
from healpix_convolution.kernels.gaussian import healpix_resolution, compute_ring


@pytest.mark.parametrize("level, expected", [
    # level = 0 --> sqrt(4*pi/(12)) --> sqrt(pi/3)
    (0, np.sqrt(np.pi/3)),
    # level = 1 --> sqrt(4*pi/(12*4)) --> sqrt(pi/12)
    (1, np.sqrt(np.pi/12))
])
def test_healpix_resolution(level, expected):
    result = healpix_resolution(level)
    np.testing.assert_allclose(result, expected)

@pytest.mark.parametrize("level, sigma, truncate, kernel_size, expected", [
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
])
def test_compute_ring(level, sigma, truncate, kernel_size, expected):
    result = compute_ring(level, sigma, truncate, kernel_size)
    assert result == expected