#!/usr/bin/env -S uv run --script
"""Plot observation coverage for whole time period"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import seaborn as sns

sns.set_theme(context="paper", style="white")
sns.set_color_codes()


DATA_DIR = Path("../get_xco2_locations/output")

FIGURES = Path("figures")

DATASET = "land_nadir"
MINIMUM_COVERAGE = 0.9


def read_obs_num(infile):
    ds = xr.open_dataset(infile)
    obs_area_frac = ds["obs_area_frac"]
    obs_num = xr.where(obs_area_frac >= MINIMUM_COVERAGE, 1, 0).sum(dim="time")
    ds.close()
    return obs_num


def create_map(ax):
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")


obs_num_summer = read_obs_num(DATA_DIR / "summer" / f"{DATASET}.nc")
obs_num_winter = read_obs_num(DATA_DIR / "winter" / f"{DATASET}.nc")

ds_wrf = xr.open_dataset("data/wrf_domain/wrfinput_d01")

lats = ds_wrf["XLAT"].isel(Time=0)
lons = ds_wrf["XLONG"].isel(Time=0)

truelat1 = ds_wrf.attrs["TRUELAT1"]
truelat2 = ds_wrf.attrs["TRUELAT2"]
stand_lon = ds_wrf.attrs["STAND_LON"]
cen_lat = ds_wrf.attrs["CEN_LAT"]
cen_lon = ds_wrf.attrs["CEN_LON"]

proj = ccrs.LambertConformal(
    central_longitude=stand_lon,
    central_latitude=cen_lat,
    standard_parallels=(truelat1, truelat2),
)

cmap = plt.cm.viridis.copy()
cmap.set_under("white")
N = obs_num_summer.data.max() + 1
colors = cmap(np.linspace(0, 1, N))
colors[0] = [1, 1, 1, 1]
cmap = mcolors.ListedColormap(colors)


# %% Plot

fig = plt.figure(figsize=(7, 4.2))
gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[18, 1],
    hspace=0.02,
    wspace=0.05,
    left=0.05,
    right=0.95,
    bottom=0.15,
    top=0.98,
)

ax = fig.add_subplot(gs[0, 0], projection=proj)
create_map(ax)

cs = ax.pcolor(
    lons,
    lats,
    obs_num_summer,
    vmin=0,
    vmax=N,
    cmap=cmap,
    transform=ccrs.PlateCarree(),
)

ax = fig.add_subplot(gs[0, 1], projection=proj)
create_map(ax)

ax.pcolor(
    lons,
    lats,
    obs_num_winter,
    vmin=0,
    vmax=N,
    cmap=cmap,
    transform=ccrs.PlateCarree(),
)

cax = fig.add_subplot(gs[1, :])
cbar = fig.colorbar(cs, cax=cax, orientation="horizontal")
cbar.set_label("Number of XCO2 observations")

ticks = np.arange(N)
cbar.set_ticks(ticks + 0.5)
cbar.set_ticklabels(ticks)

FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "xco2.png")

plt.show()
