#!/usr/bin/env python
"""Plot map of prior, true, and posterior fluxes."""

from datetime import datetime

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

import seaborn as sns

sns.set_theme("paper", "ticks")
sns.set_color_codes()


CASE = "summer"

MONTH = {
    "summer": 7,
    "winter": 2,
}[CASE]

VMAX = {
    "summer": 8e6,
    "winter": 8e5,
}[CASE]


def load_flux(infile):
    with nc.Dataset(infile) as ds:
        time_var = ds.variables["time"]
        time = nc.num2date(
            time_var[:],
            time_var.units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        fluxes = ds.variables["flux_bio_mean"][:]

    return time, fluxes


def load_total_flux(infile, mask, month):
    """This function is mainly used to reduce memory usage."""
    with nc.Dataset(infile) as ds:
        time_var = ds.variables["time"]
        time = nc.num2date(
            time_var[:],
            time_var.units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )

        months = np.array([d.month for d in time])
        idx = np.nonzero(months == month)[0]

        fluxes = ds.variables["flux_bio"][idx[0] : idx[-1] + 1]

    # Exclude grid points with no fluxes
    fluxes[:, :, mask] = np.nan

    total_fluxes = np.nansum(fluxes, axis=0)
    return total_fluxes


def mae(y, y_truth):
    return np.nansum(np.abs(y - y_truth)) / 1e6


def rmse(y, y_truth):
    return np.sqrt(np.nanmean((y - y_truth) ** 2, axis=(-1, -2))) / 1e3


def calc_corr(x, y):
    xx = x[np.isfinite(x)]
    yy = y[np.isfinite(x)]
    return np.corrcoef(xx, yy)[0, 1]


# %% Load data

time, fluxes_ref = load_flux(f"data/wp2_{CASE}/fluxes_truth.nc")
mask = fluxes_ref.sum(axis=0) == 0

total_fluxes_truth_all = load_total_flux(
    f"data/wp2_{CASE}/fluxes_truth.nc", mask=mask, month=MONTH
)

total_fluxes_prior_all = load_total_flux(
    f"data/wp2_{CASE}/fluxes_prior.nc", mask=mask, month=MONTH
)

total_fluxes_post_all = load_total_flux(
    f"data/wp2_{CASE}/fluxes_inversion.nc", mask=mask, month=MONTH
)


# %% Calculations

# Ensemble mean
total_fluxes_truth = total_fluxes_truth_all.mean(axis=0)
total_fluxes_prior = total_fluxes_prior_all.mean(axis=0)
total_fluxes_post = total_fluxes_post_all.mean(axis=0)

# Domain sum
sum_truth = total_fluxes_truth.sum()
sum_prior = total_fluxes_prior.sum()
sum_post = total_fluxes_post.sum()

sum_prior_std = total_fluxes_prior_all.sum(axis=(-1, -2)).std(ddof=1)
sum_post_std = total_fluxes_post_all.sum(axis=(-1, -2)).std(ddof=1)

# Calculate statistics
corr_prior = calc_corr(total_fluxes_prior, total_fluxes_truth)
corr_post = calc_corr(total_fluxes_post, total_fluxes_truth)

mae_prior = mae(total_fluxes_prior, total_fluxes_truth)
mae_post = mae(total_fluxes_post, total_fluxes_truth)

rmse_prior = rmse(total_fluxes_prior, total_fluxes_truth)
rmse_post = rmse(total_fluxes_post, total_fluxes_truth)


# %% Print

print(":: r")
print(f"-> prior: {corr_prior:.2f}")
print(f"-> posterior: {corr_post:.2f}")

print(":: MAE")
print(f"-> prior: {mae_prior:.1f}")
print(f"-> posterior: {mae_post:.1f}")

print(":: RMSE")
print(f"-> prior: {rmse_prior:.1f}")
print(f"-> posterior: {rmse_post:.1f}")


# %% Plot

fig = plt.figure(figsize=(8, 6))

pprop = dict(vmin=-VMAX, vmax=VMAX, cmap="PRGn_r")

ax = fig.add_subplot(1, 3, 1)
ax.set_title("Prior")
cs = ax.pcolormesh(total_fluxes_prior, **pprop)

ax = fig.add_subplot(1, 3, 2)
ax.set_title("Truth")
cs = ax.pcolormesh(total_fluxes_truth, **pprop)

height = 0.2
xmargin = 0
ymargin = 0
iax = ax.inset_axes([xmargin, 1 - height - ymargin, 0.4, 0.2])
x = np.arange(3)
y = np.array([sum_prior, sum_truth, sum_post])
yerr = np.array([sum_prior_std, np.nan, sum_post_std])

iax.bar(
    x,
    y,
    yerr=yerr,
    color=["r", "k", "b"],
    capsize=4,
    tick_label=["Prior", "Truth", "Posterior"],
)

iax.set_ylim(-16e9, 2e9)

# tprop = dict(color="w", ha="center", va="center", rotation=90)
# iax.text(x[0], y[0] / 2, "Prior", **tprop)
# iax.text(x[1], y[0] / 2, "Truth", **tprop)
# iax.text(x[2], y[0] / 2, "Posterior", **tprop)

ax = fig.add_subplot(1, 3, 3)
ax.set_title("Posterior")
cs = ax.pcolormesh(total_fluxes_post, **pprop)

# fig.colorbar(cs, extend="both")

plt.show()
