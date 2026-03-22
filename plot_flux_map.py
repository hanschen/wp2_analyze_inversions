#!/usr/bin/env -S uv run --script
"""Plot map of prior, true, and posterior fluxes."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import seaborn as sns

sns.set_theme("paper", "white")
sns.set_color_codes()


FIGURES = Path("figures")

CASE = "summer"
# CASE = "winter"

VMAX = {
    "summer": 150,
    "winter": 150,
}[CASE]


def convert(fluxes):
    """Convert from mol km-2 month-1 to gC m-2 month-1."""
    conv_factor = 12.01 * 1e-6
    return conv_factor * fluxes


def mae(y, y_truth):
    return np.nansum(np.abs(y - y_truth)) / 1e6


def rmse(y, y_truth):
    return np.sqrt(np.nanmean((y - y_truth) ** 2, axis=(-1, -2))) / 1e3


def calc_corr(x, y):
    xx = x[np.isfinite(x)]
    yy = y[np.isfinite(x)]
    return np.corrcoef(xx, yy)[0, 1]


def create_map(ax):
    # west, east, south, north
    extent = [-10, 35, 35, 71]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")


def annotate(ax, letter):
    ax.text(
        0.048,
        0.95,
        letter,
        color="k",
        fontsize=12,
        ha="left",
        va="top",
        bbox=dict(facecolor="w", edgecolor="k"),
        transform=ax.transAxes,
    )


# %% Load data

fluxes_truth = np.load("output/mon_truth.npy")
fluxes_prior = np.load("output/mon_prior.npy")
fluxes_post = np.load("output/mon_post.npy")

fluxes_truth = convert(fluxes_truth)
fluxes_prior = convert(fluxes_prior)
fluxes_post = convert(fluxes_post)

with nc.Dataset("data/wrf_domain/geo_em.d01.nc") as ds:
    lats = ds.variables["XLAT_M"][0]
    lons = ds.variables["XLONG_M"][0]


# %% Calculations

# Ensemble mean
fluxes_mean_truth = fluxes_truth.mean(axis=0)
fluxes_mean_prior = fluxes_prior.mean(axis=0)
fluxes_mean_post = fluxes_post.mean(axis=0)

# Domain sum
sum_truth = np.nansum(fluxes_mean_truth)
sum_prior = np.nansum(fluxes_mean_prior)
sum_post = np.nansum(fluxes_mean_post)

sum_prior_std = np.nansum(fluxes_prior, axis=(-1, -2)).std(ddof=1)
sum_post_std = np.nansum(fluxes_post, axis=(-1, -2)).std(ddof=1)

# Calculate statistics
corr_prior = calc_corr(fluxes_mean_prior, fluxes_mean_truth)
corr_post = calc_corr(fluxes_mean_post, fluxes_mean_truth)

mae_prior = mae(fluxes_mean_prior, fluxes_mean_truth)
mae_post = mae(fluxes_mean_post, fluxes_mean_truth)

rmse_prior = rmse(fluxes_mean_prior, fluxes_mean_truth)
rmse_post = rmse(fluxes_mean_post, fluxes_mean_truth)


# %% Print

print(":: corr")
print(f"-> prior: {corr_prior:.2f}")
print(f"-> posterior: {corr_post:.2f}")

print(":: MAE")
print(f"-> prior: {mae_prior:.1f}")
print(f"-> posterior: {mae_post:.1f}")
print(f"-> relative reduction: {(1 - mae_post / mae_prior) * 100:.1f}%")

print(":: RMSE")
print(f"-> prior: {rmse_prior:.1f}")
print(f"-> posterior: {rmse_post:.1f}")
print(f"-> relative reduction: {(1 - rmse_post / rmse_prior) * 100:.1f}%")


# %% Plot

fig = plt.figure(figsize=(7, 2.8))
gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[18, 1],
    hspace=0.01,
    wspace=0.05,
    left=0.01,
    right=0.99,
    bottom=0.2,
    top=0.95,
)

pprop = dict(
    vmin=-VMAX, vmax=VMAX, cmap="PRGn_r", transform=ccrs.PlateCarree()
)

ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
annotate(ax, "a")
ax.set_title("Prior")
create_map(ax)
cs = ax.pcolormesh(lons, lats, fluxes_mean_prior, **pprop)

ax = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
annotate(ax, "b")
ax.set_title("Truth")
create_map(ax)
cs = ax.pcolor(lons, lats, fluxes_mean_truth, **pprop)

# width = 0.3
# height = 0.2
# xmargin = 0
# ymargin = 0
# iax = ax.inset_axes([xmargin, 1 - height - ymargin, width, height])
# x = np.arange(3)
# y = np.array([sum_prior, sum_truth, sum_post])
# yerr = np.array([sum_prior_std, np.nan, sum_post_std])
# iax.bar(
#     x,
#     y,
#     yerr=yerr,
#     color=["r", "k", "b"],
#     capsize=4,
#     # tick_label=["Prior", "Truth", "Posterior"],
# )
# iax.set_xticks([])
# iax.set_yticks([])
# tprop = dict(color="w", ha="center", va="center", rotation=90)
# iax.text(x[0], y[0] / 2, "Prior", **tprop)
# iax.text(x[1], y[0] / 2, "Truth", **tprop)
# iax.text(x[2], y[0] / 2, "Posterior", **tprop)

ax = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
annotate(ax, "c")
ax.set_title("Posterior")
create_map(ax)
cs = ax.pcolor(lons, lats, fluxes_mean_post, **pprop)

cax = fig.add_subplot(gs[1, :])
cbar = fig.colorbar(cs, cax=cax, orientation="horizontal", extend="both")
cbar.set_label("Net ecosystem exchange (gC m$^{-2}$ month$^{-1}$)")

FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / f"map_{CASE}.png")

plt.show()
