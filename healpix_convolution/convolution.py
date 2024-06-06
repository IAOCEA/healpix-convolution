import dask.array as da
import opt_einsum


def convolve(arr, kernel, **kwargs):
    """convolve an array using a pre-computed sparse kernel matrix

    Parameters
    ----------
    arr : array-like
        The data to convolve. The dimension to contract must be the last axis.
    kernel : array-like
        2-dimensional sparse matrix. Columns is the original pixels, rows the output pixels.
    axis : int, optional
        The axis along which to apply the convolution
    """
    if isinstance(arr, da.Array) and not isinstance(kernel, da.Array):
        kernel = da.from_array(kernel)

    return opt_einsum.contract("...a,ba->...b", arr, kernel, **kwargs)
