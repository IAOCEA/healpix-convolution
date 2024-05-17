import healpy as hp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import healpix_convolution.neighbours as nb

try:
    import dask.array as da

    has_dask = True
except ImportError:
    has_dask = False
    da = None

requires_dask = pytest.mark.skipif(not has_dask, reason="needs dask.array")


@given(st.integers(min_value=0, max_value=300))
def test_generate_offsets(ring):
    kernel_size = 2 * ring + 1
    kernel = np.zeros(shape=(kernel_size, kernel_size))

    for x, y in nb.generate_offsets(ring):
        kernel[x + ring, y + ring] = 1

    assert np.sum(kernel) == kernel_size**2


@pytest.mark.parametrize("resolution", [1, 2, 4, 6])
@pytest.mark.parametrize("indexing_scheme", ["ring", "nested"])
@pytest.mark.parametrize("dask", [pytest.param(True, marks=requires_dask), False])
def test_neighbours_ring1_manual(resolution, indexing_scheme, dask):
    if dask:
        xp = da
    else:
        xp = np

    cell_ids = xp.arange(12 * 4**resolution)

    actual = nb.neighbours(
        cell_ids, resolution=resolution, indexing_scheme=indexing_scheme, ring=1
    )

    nside = 2**resolution
    nest = indexing_scheme == "nested"
    expected = hp.get_all_neighbours(nside, np.asarray(cell_ids), nest=nest).T

    actual_ = np.asarray(actual[:, 1:])

    np.testing.assert_equal(actual_, expected)


@given(st.integers(min_value=1, max_value=7), st.sampled_from(["ring", "nested"]))
def test_neighbours_ring1(resolution, indexing_scheme):
    cell_ids = np.arange(12 * 4**resolution)

    actual = nb.neighbours(
        cell_ids, resolution=resolution, indexing_scheme=indexing_scheme, ring=1
    )
    nside = 2**resolution
    nest = indexing_scheme == "nested"
    expected = hp.get_all_neighbours(nside, cell_ids, nest=nest).T

    np.testing.assert_equal(actual[:, 1:], expected)
