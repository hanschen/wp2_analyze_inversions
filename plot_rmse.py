#!/usr/bin/env -S uv run --script
"""Plot posterior flux and parameter RMSE relative to prior over time."""

from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import seaborn as sns

sns.set_theme("paper", "ticks")
sns.set_color_codes()

CASE = "summer"
# CASE = "winter"

CYCLE_START = {
    "summer": datetime(2021, 6, 17, 0, 0, 0),
    "winter": datetime(2021, 1, 18, 0, 0, 0),
}[CASE]

SPINUP_END = {
    "summer": datetime(2021, 7, 1, 0, 0, 0),
    "winter": datetime(2021, 2, 1, 0, 0, 0),
}[CASE]

ANNOTATION = {
    "summer": "a",
    "winter": "b",
}[CASE]

SHOW_LEGEND = {
    "summer": False,
    "winter": True,
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


def load_parameter(infile):
    with nc.Dataset(infile) as ds:
        time_var = ds.variables["time"]
        time = nc.num2date(
            time_var[:],
            time_var.units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        parameters = ds.variables["parameter_bio_mean"][:]

    return time, parameters


def hourly2daily(y):
    daily_dim = (-1, 24) + y.shape[1:]
    return y.reshape(daily_dim).mean(axis=1)


def rmse(y, y_truth):
    return np.sqrt(np.nanmean((y - y_truth) ** 2, axis=(-1, -2)))


def rer(rmse_post, rmse_prior):
    return (1 - rmse_post / rmse_prior) * 100


# %% Load data

time, fluxes_truth = load_flux(f"data/wp2_{CASE}/fluxes_truth.nc")
fluxes_prior = load_flux(f"data/wp2_{CASE}/fluxes_prior.nc")[1]
fluxes_post = load_flux(f"data/wp2_{CASE}/fluxes_inversion.nc")[1]

parameters_truth = load_parameter(f"data/wp2_{CASE}/parameters_truth.nc")[1]
parameters_prior = np.ones_like(parameters_truth)
parameters_post = load_parameter(f"data/wp2_{CASE}/parameters_inversion.nc")[1]


# %% Calculations

# Exclude grid points with no fluxes
mask = fluxes_truth == 0

fluxes_truth[mask] = np.nan
fluxes_prior[mask] = np.nan
fluxes_post[mask] = np.nan

parameters_truth[mask] = np.nan
parameters_prior[mask] = np.nan
parameters_post[mask] = np.nan

# Exclude last timestep
time = time[:-1]

fluxes_truth = fluxes_truth[:-1]
fluxes_prior = fluxes_prior[:-1]
fluxes_post = fluxes_post[:-1]

parameters_truth = parameters_truth[:-1]
parameters_prior = parameters_prior[:-1]
parameters_post = parameters_post[:-1]

# Calculate daily values
time_daily = time[::24]

fluxes_daily_truth = hourly2daily(fluxes_truth)
fluxes_daily_prior = hourly2daily(fluxes_prior)
fluxes_daily_post = hourly2daily(fluxes_post)

parameters_daily_truth = hourly2daily(parameters_truth)
parameters_daily_prior = hourly2daily(parameters_prior)
parameters_daily_post = hourly2daily(parameters_post)

# Calculate RMSEs and RERs
rmse_prior = rmse(fluxes_daily_prior, fluxes_daily_truth)
rmse_post = rmse(fluxes_daily_post, fluxes_daily_truth)

rmse_prior_parameters = rmse(parameters_daily_prior, parameters_daily_truth)
rmse_post_parameters = rmse(parameters_daily_post, parameters_daily_truth)

rer_fluxes = rer(rmse_post, rmse_prior)
rer_parameters = rer(rmse_post_parameters, rmse_prior_parameters)

start = np.nonzero(time_daily == SPINUP_END)[0].item()
rer_fluxes_mean = rer_fluxes[start:].mean()
rer_parameters_mean = rer_parameters[start:].mean()

print(":: RER")
print(f"-> fluxes: {rer_fluxes_mean:.1f}%")
print(f"-> parameters: {rer_parameters_mean:.1f}%")


# %% Plot

fig = plt.figure(figsize=(3.5, 3))
ax = fig.add_subplot(111)

ax.plot(
    time_daily,
    rer_fluxes,
    color="g",
    linewidth=1.5,
    label="Fluxes",
)

ax.plot(
    time_daily,
    rer_parameters,
    color="b",
    linewidth=1.5,
    label="Parameters",
)

ax.fill_between(
    [CYCLE_START, SPINUP_END],
    0,
    1,
    facecolor="0.9",
    edgecolor="0.8",
    hatch="/",
    transform=ax.get_xaxis_transform(),
    # label="Spinup",
)

ax.text(
    0.04,
    0.96,
    ANNOTATION,
    color="k",
    fontsize=12,
    ha="left",
    va="top",
    bbox=dict(facecolor="w", edgecolor="k"),
    transform=ax.transAxes,
)

# Mean error reduction
# ax.plot(
#     [SPINUP_END, time_daily[-1]],
#     [rer_fluxes_mean, rer_fluxes_mean],
#     color="g",
#     linewidth=1.5,
#     linestyle="--",
# )

# ax.plot(
#     [SPINUP_END, time_daily[-1]],
#     [rer_parameters_mean, rer_parameters_mean],
#     color="b",
#     linewidth=1.5,
#     linestyle="--",
# )

ax.set_xticks(time_daily[::7])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# for label in ax.get_xticklabels(which="major"):
#     label.set(rotation=30, horizontalalignment="right")
ax.set_xlim(CYCLE_START, time_daily[-1])
ax.set_ylim(0, 100)

if SHOW_LEGEND:
    ax.legend(loc="upper right")
ax.set_xlabel("Date")
ax.set_ylabel("Relative error reduction (%)")

fig.tight_layout()
fig.savefig(f"output/rmse_{CASE}.png")

plt.show()
