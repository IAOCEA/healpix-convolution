# healpix-convolution: convolution on the healpix grid

```{toctree}
---
maxdepth: 3
caption: User guide
hidden: true
---
Convolution of bare arrays <user-guide/array>
Convolution of `xarray` objects <user-guide/xarray>
```

```{toctree}
---
maxdepth: 3
caption: Reference guide
hidden: true
---

API Reference <api>
```

Convolving on the healpix grid has traditionally been done using spherical harmonics, and thus in the frequency domain.

However, for small kernels this has the non-negligible overhead of transforming from / to spherical harmonics, and thus tends to be slow.

To resolve this, this package instead constructs a sparse 2-dimensional matrix as the kernel, which can then be sparse-multiplied with the data. Additionally, to allow convolving subdomains it also provides tools to pad the data.
