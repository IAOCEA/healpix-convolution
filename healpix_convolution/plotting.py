import cartopy.crs as ccrs
import cartopy.feature
import healpy as hp
import numpy as np


def plot_healpix(
    data, cell_ids, *, ax, resolution, cmap="viridis", xsize=1200, title=None, **kwargs
):
    nside = 2**resolution

    ysize = xsize // 2
    full_lat = np.linspace(-90, 90, ysize)
    full_lon = np.linspace(-180, 180, xsize)
    grid_lat, grid_lon = np.meshgrid(full_lat, full_lon)
    pix = hp.ang2pix(nside, grid_lon, grid_lat, lonlat=True, nest=True)

    full_map = np.full((12 * nside**2,), fill_value=np.nan)
    full_map[cell_ids] = data
    grid_map = full_map[pix]

    row_mask = np.logical_not(np.all(np.isnan(grid_map), axis=1))
    col_mask = np.logical_not(np.all(np.isnan(grid_map), axis=0))

    subdomain = grid_map[row_mask, :][:, col_mask]
    lon = grid_lon[row_mask, :][:, col_mask]
    lat = grid_lat[row_mask, :][:, col_mask]
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=5)

    if title is not None:
        ax.set_title(title)

    return ax.pcolormesh(
        lon, lat, subdomain, cmap=cmap, transform=ccrs.PlateCarree(), **kwargs
    )


def xr_plot_healpix(arr, *, ax, cmap="viridis", xsize=1200, title=None, **kwargs):
    cell_ids_ = arr["cell_ids"]
    cell_ids = cell_ids_.data

    params = cell_ids_.attrs
    resolution = params["resolution"]

    return plot_healpix(
        arr.data,
        cell_ids,
        resolution=resolution,
        ax=ax,
        cmap=cmap,
        xsize=xsize,
        title=title,
        **kwargs,
    )
