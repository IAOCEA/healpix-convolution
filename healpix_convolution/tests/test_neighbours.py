import cdshealpix
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from xdggs import HealpixInfo

from healpix_convolution.neighbours import neighbours

try:
    import dask.array as da

    has_dask = True
except ImportError:  # pragma: no cover
    has_dask = False
    da = None

requires_dask = pytest.mark.skipif(not has_dask, reason="needs dask.array")


@pytest.mark.parametrize("resolution", [1, 2, 4, 6])
@pytest.mark.parametrize(
    "indexing_scheme",
    [
        pytest.param("ring", marks=pytest.mark.skip(reason="not implemented yet")),
        "nested",
    ],
)
@pytest.mark.parametrize("dask", [pytest.param(True, marks=requires_dask), False])
def test_neighbours_ring1_manual(resolution, indexing_scheme, dask):
    if dask:
        xp = da
    else:
        xp = np

    # no other scheme supported
    reference_neighbours = cdshealpix.nested.neighbours

    grid_info = HealpixInfo(level=resolution, indexing_scheme=indexing_scheme)

    cell_ids = xp.arange(12 * 4**resolution)

    actual = neighbours(cell_ids, grid_info=grid_info, ring=1)

    expected = reference_neighbours(np.asarray(cell_ids), depth=grid_info.level)

    np.testing.assert_equal(np.sort(actual, axis=1), np.sort(expected, axis=1))


@given(st.integers(min_value=1, max_value=7), st.sampled_from(["nested"]))
def test_neighbours_ring1(resolution, indexing_scheme):
    # no other scheme supported
    reference_neighbours = cdshealpix.nested.neighbours

    cell_ids = np.arange(12 * 4**resolution)

    grid_info = HealpixInfo(level=resolution, indexing_scheme=indexing_scheme)
    actual = neighbours(cell_ids, grid_info=grid_info, ring=1)

    expected = reference_neighbours(cell_ids, depth=grid_info.level)

    np.testing.assert_equal(np.sort(actual, axis=1), np.sort(expected, axis=1))
