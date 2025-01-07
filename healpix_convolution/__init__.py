from importlib.metadata import version

from healpix_convolution import kernels
from healpix_convolution.convolution import convolve
from healpix_convolution.distances import angular_distances
from healpix_convolution.neighbours import neighbours
from healpix_convolution.padding import Padding, pad  # noqa: F401

try:
    __version__ = version("healpix_convolution")
except Exception:  # pragma: no cover
    __version__ = "9999"

__all__ = ["kernels", "angular_distances", "neighbours", "pad", "convolve"]
