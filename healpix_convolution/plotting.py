import cartopy.crs as ccrs
import cartopy.feature
import numpy as np


def plot_healpix(
    data,
    cell_ids,
    *,
    ax,
    grid_info,
    cmap="viridis",
    xsize=1200,
    title=None,
    features=("coastlines", "land"),
    **kwargs,
):
    nside = grid_info.nside

    ysize = xsize // 2
    full_lat = np.linspace(-90, 90, ysize)
    full_lon = np.linspace(-180, 180, xsize)
    grid_lat, grid_lon = np.meshgrid(full_lat, full_lon)
    pix = grid_info.geographic2cell_ids(lon=grid_lon, lat=grid_lat)

    full_map = np.full((12 * nside**2,), fill_value=np.nan)
    full_map[cell_ids] = data
    grid_map = full_map[pix]

    row_mask = np.logical_not(np.all(np.isnan(grid_map), axis=1))
    col_mask = np.logical_not(np.all(np.isnan(grid_map), axis=0))

    subdomain = grid_map[row_mask, :][:, col_mask]
    lon = grid_lon[row_mask, :][:, col_mask]
    lat = grid_lat[row_mask, :][:, col_mask]
    if "coastlines" in features:
        ax.coastlines()

    if "land" in features:
        ax.add_feature(cartopy.feature.LAND, zorder=5)

    if title is not None:
        ax.set_title(title)

    return ax.pcolormesh(
        lon, lat, subdomain, cmap=cmap, transform=ccrs.PlateCarree(), **kwargs
    )


def xr_plot_healpix(
    arr,
    *,
    ax,
    cmap="viridis",
    xsize=1200,
    title=None,
    features=("coastlines", "land"),
    **kwargs,
):
    cell_ids_ = arr["cell_ids"]
    cell_ids = cell_ids_.data

    return plot_healpix(
        arr.data,
        cell_ids,
        grid_info=cell_ids_.dggs.grid_info,
        ax=ax,
        cmap=cmap,
        xsize=xsize,
        title=title,
        features=features,
        **kwargs,
    )
