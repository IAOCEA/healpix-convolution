from importlib.metadata import version

try:
    __version__ = version("healpix_indexing")
except Exception:
    __version__ = "9999"
