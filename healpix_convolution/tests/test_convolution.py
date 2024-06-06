import hypothesis
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
import sparse
from hypothesis import given, settings

from healpix_convolution import convolution


@pytest.fixture
def rolling_mean_kernel():
    kernel = (
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

    return sparse.COO.from_numpy(kernel, fill_value=0)


@given(
    data=npst.arrays(
        shape=st.sampled_from([(5,), (10, 5)]),
        # TODO: figure out how to deal with floating point values
        dtype=st.sampled_from(["int16", "int32", "int64"]),
    ),
)
@settings(
    deadline=1000,
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
)
def test_numpy_convolve(data, rolling_mean_kernel):
    kernel = rolling_mean_kernel
    actual = convolution.convolve(data, kernel)

    padding = [(0, 0)] * (data.ndim - 1) + [(1, 1)]
    padded = np.pad(data, padding, mode="wrap")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 3, axis=-1)
    expected = np.mean(windows, axis=-1)

    np.testing.assert_allclose(actual, expected)
