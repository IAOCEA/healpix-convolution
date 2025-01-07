# Convolution of `xarray` objects

```{jupyter-execute}
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xdggs
from xdggs.healpix import HealpixInfo

import healpix_convolution.xarray as hc
from healpix_convolution.plotting import xr_plot_healpix as plot_healpix

rng = np.random.default_rng(seed=0)
```

## Global convolution

We first have to define the data we want to convolve:

```{jupyter-execute}
grid_info = HealpixInfo(level=3, indexing_scheme="nested")

cell_ids = np.arange(12 * 4**grid_info.level)
data1 = rng.normal(size=cell_ids.shape)
data2 = rng.normal(size=cell_ids.shape)

ds = xr.Dataset(
    {"data1": ("cells", data1), "data2": ("cells", data2)},
    coords={"cell_ids": ("cells", cell_ids, grid_info.to_dict())},
).pipe(xdggs.decode)
ds
```

And configure the kernel:

```{jupyter-execute}
kernel = hc.kernels.gaussian_kernel(
    ds["cell_ids"], sigma=0.5, kernel_size=5, weights_threshold=1e-8
)
kernel
```

The convolution is then just:

```{jupyter-execute}
convolved = hc.convolve(ds, kernel).pipe(xdggs.decode)
convolved
```

And we can compare the two variables:

```{jupyter-execute}
---
stderr: true
---

fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(14, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_healpix(ds["data1"], ax=axes[0, 0], title="original1", features=["coastlines"])
plot_healpix(
    convolved["data1"], ax=axes[0, 1], title="convolved1", features=["coastlines"]
)
plot_healpix(ds["data2"], ax=axes[1, 0], title="original2", features=["coastlines"])
plot_healpix(
    convolved["data2"], ax=axes[1, 1], title="convolved2", features=["coastlines"]
)
fig.subplots_adjust(hspace=-0.15, wspace=0.1)
```

## Regional convolution

Similarly to above, we also first have to define the data:

```{jupyter-execute}
grid_info = HealpixInfo(level=4, indexing_scheme="nested")

cell_ids = 4 * 4**grid_info.level + np.arange(4**grid_info.level)
data1 = rng.normal(size=cell_ids.shape)
data2 = rng.normal(size=cell_ids.shape)

ds = xr.Dataset(
    {"data1": ("cells", data1), "data2": ("cells", data2)},
    coords={"cell_ids": ("cells", cell_ids, grid_info.to_dict())},
).pipe(xdggs.decode)
ds
```

and we can define the kernel:

```{jupyter-execute}
kernel = hc.kernels.gaussian_kernel(
    ds["cell_ids"], sigma=0.5, kernel_size=5, weights_threshold=1e-8
)
kernel
```

However, since we now have boundaries we have to pad the data before convolving:

```{jupyter-execute}
padding = hc.pad(
    ds["cell_ids"],
    ring=kernel.attrs["ring"],
    mode="constant",
    constant_value=0,
)
padded_ds = padding.apply(ds)
padded_ds
```

After that, convolving works as before (but on the padded dataset):

```{jupyter-execute}
convolved = hc.convolve(padded_ds, kernel).pipe(xdggs.decode)
convolved
```

Again, we can look at the result:

```{jupyter-execute}
---
stderr: true
---

fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(14, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_healpix(
    ds["data1"], ax=axes[0, 0], title="original1", features=["coastlines"], xsize=1000
)
plot_healpix(
    convolved["data1"],
    ax=axes[0, 1],
    title="convolved1",
    features=["coastlines"],
    xsize=1000,
)
plot_healpix(
    ds["data2"], ax=axes[1, 0], title="original2", features=["coastlines"], xsize=1000
)
plot_healpix(
    convolved["data2"],
    ax=axes[1, 1],
    title="convolved2",
    features=["coastlines"],
    xsize=1000,
)
fig.subplots_adjust(hspace=0.1, wspace=-0.55)
```
