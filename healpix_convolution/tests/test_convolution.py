import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import sparse
from hypothesis import given

from healpix_convolution import convolution


@given(
    npst.arrays(
        shape=st.sampled_from([(5,), (10, 5)]),
        dtype=st.sampled_from(
            ["float64", "float32", "int8", "int16", "int32", "int64"]
        ),
    )
)
def test_numpy_convolve(data):
    kernel = sparse.COO.from_numpy(
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

    actual = convolution.convolve(data, kernel)

    padding = [(0, 0)] * (data.ndim - 1) + [(1, 1)]
    padded = np.pad(data, padding, mode="wrap")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 3, axis=-1)
    expected = np.mean(windows, axis=-1)

    np.testing.assert_allclose(actual, expected)
