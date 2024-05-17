from importlib.metadata import version

try:
    __version__ = version("healpix_convolution")
except Exception:
    __version__ = "9999"
