#!/usr/bin/env -S uv run --script
"""Calculate monthly total fluxes."""

from pathlib import Path

import netCDF4 as nc
import numpy as np

OUTPUT = Path("output")

CASES = [
    "summer",
    "winter",
]

MONTHS = {
    "summer": 7,
    "winter": 2,
}


def load_flux(infile):
    with nc.Dataset(infile) as ds:
        fluxes = ds.variables["flux_bio_mean"][:]

    return time, fluxes


def load_monthly_flux(infile, month, mask=None):
    with nc.Dataset(infile) as ds:
        ds.set_always_mask(False)
        time_var = ds.variables["time"]
        time = nc.num2date(
            time_var[:],
            time_var.units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )

        months = np.array([d.month for d in time])
        idx = np.nonzero(months == month)[0]
        start = idx[0]
        end = idx[-1]

        print(time[start], time[end])
        fluxes = ds.variables["flux_bio"][idx[0] : idx[-1] + 1]

    if mask is not None:
        # Exclude grid points with no fluxes
        fluxes[:, :, mask] = np.nan

    monthly_fluxes = np.sum(fluxes, axis=0)
    return monthly_fluxes


for case in CASES:
    with nc.Dataset(f"data/wp2_{case}/fluxes_truth.nc") as ds:
        fluxes_ref = ds.variables["flux_bio_mean"][:]
    mask = fluxes_ref.sum(axis=0) == 0

    monthly_fluxes_truth = load_monthly_flux(
        f"data/wp2_{case}/fluxes_truth.nc",
        month=MONTHS[case],
        mask=mask,
    )

    monthly_fluxes_prior = load_monthly_flux(
        f"data/wp2_{case}/fluxes_prior.nc",
        month=MONTHS[case],
        mask=mask,
    )

    monthly_fluxes_post = load_monthly_flux(
        f"data/wp2_{case}/fluxes_inversion.nc",
        month=MONTHS[case],
        mask=mask,
    )

    output = OUTPUT / case
    output.mkdir(exist_ok=True)
    np.save(output / "mon_truth", monthly_fluxes_truth)
    np.save(output / "mon_prior", monthly_fluxes_prior)
    np.save(output / "mon_post", monthly_fluxes_post)
