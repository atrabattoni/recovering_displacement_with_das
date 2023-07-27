import os
from glob import glob

import cartopy.crs as ccrs
import numpy as np
import obspy
import pandas as pd
import scipy.signal as sp
import xarray as xr
import xdas.signal as xp


def das_relative_dist(coords, tr):
    lon = coords["longitude"].values
    lat = coords["latitude"].values
    crs = ccrs.AzimuthalEquidistant(lon[0], lat[0])
    xe, ye = crs.transform_point(
        tr.stats.sac["evlo"], tr.stats.sac["evla"], ccrs.Geodetic()
    )
    re = xe + 1j * ye
    return re


# parameters
duration = 20  # s noise duration selection
snr_thresh = 10  # Signal to Noise Ratio threshold
min_ch = 30  # Minimum number of channel to process the event

# parameters WA resp removal
freq = 25.0
corners = 4
taper = 0.1
paz = {
    "sensitivity": 2800,
    "zeros": [0j, 0j],
    "gain": 1,
    "poles": [-6.2832 - 4.7124j, -6.2832 + 4.7124j],
}

# Colliano fiber's segment
limit_values = [59, 357, 926]
limit_names = ["A", "B", "C"]
limits = dict(zip(limit_names, limit_values))

# Read DAS data
dasfnames = sorted(glob("data/irpinia/das/*.nc"))
coords = xr.open_dataset("data/irpinia/fiber_dgnss.nc")

catalog = []
magnitudes = {}
for dasfname in dasfnames:
    # Load DAS data
    event = os.path.splitext(os.path.split(dasfname)[-1])[0]

    # load, filter and convert to Wood-Anderson displacement in millimeters.
    strainrate = xr.open_dataarray(dasfname)
    strainrate = strainrate.sel(offset=slice(limits["A"], limits["C"]))
    strainrate = xp.detrend(strainrate, type="linear", dim="time")
    strainrate = xp.taper(strainrate, ("tukey", taper), dim="time")
    strainrate = xp.iirfilter(strainrate, freq, "lowpass", corners)
    strain = xp.integrate(strainrate, dim="time")
    deformation = xp.integrate(strain, dim="offset")
    displacement = xp.sliding_mean_removal(deformation, wlen=250.0, dim="offset")
    displacement = 1e-6 * displacement
    b, a = sp.zpk2tf(paz["zeros"], paz["poles"], paz["gain"])
    b, a = sp.bilinear(b, a, fs=1.0 / xp.get_sample_spacing(displacement, "time"))
    displacement = displacement.copy(data=sp.lfilter(b, a, displacement.data, axis=0))

    # SNR based event and channel selection
    starttime = displacement["time"][0].values
    endtime = starttime + np.timedelta64(duration, "s")
    noise = displacement.sel(time=slice(starttime, endtime)).std("time")
    peak = np.abs(displacement).max("time")
    snr = peak / noise
    mask = snr > snr_thresh
    if np.count_nonzero(mask) < min_ch:
        continue
    peak[{"offset": ~mask}] = np.nan

    # hypocentral distance estimation
    st = obspy.read(os.path.join("data/irpinia/sac", event, "*.C0[4].*.sac"))
    tr = st[0]
    re = das_relative_dist(coords, tr)
    distance_to_das = np.abs(re) / 1000  # Distances are in km
    hyp = np.sqrt(tr.stats.sac["evdp"] ** 2 + distance_to_das**2)

    # local magnitude evaluation from Bobbio et al. (2009)
    A = 2800 * peak
    magnitude = np.log10(A) + 1.79 * np.log10(hyp) - 0.58

    catalog.append(
        {
            "event": event,
            "magnitude": np.nanmedian(magnitude),
            "smad": 1.4826 * np.nanmedian(np.abs(magnitude - np.nanmedian(magnitude))),
        }
    )
    magnitudes[event] = magnitude

catalog = pd.DataFrame.from_records(catalog, index="event")
catalog.to_csv("results/das_catalog.csv")

magnitudes = xr.Dataset(magnitudes)
magnitudes.to_netcdf("results/das_magnitudes.nc")
