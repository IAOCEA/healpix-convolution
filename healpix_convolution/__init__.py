from importlib.metadata import version

from healpix_convolution import kernels  # noqa: F401
from healpix_convolution.distances import angular_distances  # noqa: F401
from healpix_convolution.neighbours import neighbours  # noqa: F401

try:
    __version__ = version("healpix_convolution")
except Exception:  # pragma: no cover
    __version__ = "9999"
