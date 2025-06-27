# Convolution of bare arrays

As a basic building block, the API for bare arrays exists.

Before getting into how this works, we first needs to import a few things:

```{jupyter-execute}
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from xdggs.healpix import HealpixInfo

import healpix_convolution as hc
from healpix_convolution.plotting import plot_healpix

rng = np.random.default_rng(seed=0)
```

## Global convolution

We first define the grid and the data to convolve:

```{jupyter-execute}
grid_info = HealpixInfo(level=3, indexing_scheme="nested")

cell_ids = np.arange(12 * 4**grid_info.level)
data = rng.normal(size=cell_ids.shape)
```

We also need a kernel:

```{jupyter-execute}
_, kernel = hc.kernels.gaussian_kernel(
    cell_ids,
    grid_info=grid_info,
    sigma=0.5,
    kernel_size=5,
    weights_threshold=1e-8,
)
kernel
```

With all that, convolution is simply:

```{jupyter-execute}
convolved = hc.convolve(data, kernel)
convolved.shape
```

On a map:

```{jupyter-execute}
---
stderr: true
---
fig, axes = plt.subplots(
    ncols=2, figsize=(16, 12), subplot_kw={"projection": ccrs.Robinson()}
)

plot_healpix(data, cell_ids, ax=axes[0], grid_info=grid_info, features=["coastlines"])
plot_healpix(
    convolved,
    cell_ids,
    ax=axes[1],
    grid_info=grid_info,
    features=["coastlines"],
)
```

## Regional convolution

Similarly to above, we also first have to define the data:

```{jupyter-execute}
grid_info = HealpixInfo(level=4, indexing_scheme="nested")

cell_ids = 4 * 4**grid_info.level + np.arange(4**grid_info.level)
data = rng.normal(size=cell_ids.shape)
```

and we can immediately define the kernel:

```{jupyter-execute}
kernel_size = 5
_, kernel = hc.kernels.gaussian_kernel(
    cell_ids,
    grid_info=grid_info,
    sigma=0.5,
    kernel_size=kernel_size,
    weights_threshold=1e-10,
)
kernel
```

However, since we now have boundaries we have to pad the data before convolving:

```{jupyter-execute}
padded_data = hc.pad(
    cell_ids,
    grid_info=grid_info,
    ring=kernel_size // 2,
    mode="constant",
    constant_value=0,
).apply(data)
padded_data.shape
```

After that, convolving works as before (but on the padded data):

```{jupyter-execute}
convolved = hc.convolve(padded_data, kernel)
```

Again, we can look at the result:

```{jupyter-execute}
---
stderr: true
---
fig, axes = plt.subplots(
    ncols=2, figsize=(16, 12), subplot_kw={"projection": ccrs.Robinson()}
)

plot_healpix(
    data,
    cell_ids,
    ax=axes[0],
    grid_info=grid_info,
    features=["coastlines"],
    xsize=3000,
)
plot_healpix(
    convolved,
    cell_ids,
    ax=axes[1],
    grid_info=grid_info,
    features=["coastlines"],
    xsize=3000,
)
```
