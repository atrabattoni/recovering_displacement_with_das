import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import CubicSpline

import numpy as np
import obspy
import pandas as pd
import scipy.signal as sp
import xarray as xr
import xdas.signal as xp
from obspy.signal.rotate import rotate_zne_lqt

# Parameters
freqmin = 2.5
freqmax = 15.0
gain = 1e-9
L = 250.0
markers = dict(zip(["A", "B", "C", "D", "E"], [1048.8, 1111.2, 1411.2, 1521.6, 1670.4]))
limits = list(markers.values())


# Load data
dtm = xr.open_dataarray("data/stromboli/dtm.nc")
fiber_all = pd.read_csv("data/stromboli/fiber.csv").set_index("offset").to_xarray()
stations = (
    pd.read_csv("data/stromboli/stations.csv", dtype={"station": str})
    .set_index("station")
    .to_xarray()
)
strainrate = xr.open_dataarray("data/stromboli/das.nc")
st = obspy.read("data/stromboli/mseed/*.mseed")


# Compute cable 3D direction
def central_diff(xarr):
    data = xarr.values
    data = np.pad(data, 1)  # constant_values=np.nan
    data = data[2:] - data[:-2]
    return xarr.copy(data=data)


delta = fiber_all.map(central_diff)
delta = delta / np.sqrt(np.square(delta).to_array("dimension").sum("dimension"))
fiber_all["orientation"] = np.rad2deg(np.arctan2(delta["x"], delta["y"])) % 360
fiber_all["tilt"] = np.rad2deg(np.arccos(delta["z"])) % 360

# Get subsection
fiber = fiber_all.interp_like(strainrate)

# Find closest channels
distance = np.sqrt(
    np.square(fiber.sel(offset=slice(None, 1600)) - stations)
    .to_array("dimension")
    .sum("dimension")
)
indices = distance.argmin("offset")
channels = fiber.isel(offset=indices)

# Local time
starttime = obspy.UTCDateTime(str(strainrate["time"][0].values))
endtime = obspy.UTCDateTime(str(strainrate["time"][-1].values))
strainrate["time"] = (
    strainrate["time"] - strainrate["time"][0]
).values / np.timedelta64(1, "s")

# Process DAS data
strainrate *= gain
strainrate = xp.iirfilter(strainrate, [freqmin, freqmax], "bandpass", dim="time")
deformationrate = xp.integrate(strainrate, dim="offset")

# Align seismometers traces
st.trim(starttime, endtime)

# Band pass
st.filter(type="bandpass", freqmin=freqmin, freqmax=freqmax)

# Project along the cable direction
traces = {}
for station in stations["station"].values:
    orientation = channels["orientation"].sel(station=station)
    tilt = channels["tilt"].sel(station=station)
    data, _, _ = rotate_zne_lqt(
        st.select(station=station, channel="*Z")[0].data,
        st.select(station=station, channel="*N")[0].data,
        st.select(station=station, channel="*E")[0].data,
        orientation,
        tilt,
    )
    data = sp.decimate(data, 5, ftype="fir")
    traces[station] = data


# trace comparison
wlen = np.arange(50, 1000 + 10, 10)
shape = (len(wlen), len(stations["station"]))
amplitude_ratio = np.zeros(shape)
corrcoef = np.zeros(shape)

for i, L in enumerate(wlen):
    velocity = xp.sliding_mean_removal(deformationrate, wlen=L, dim="offset", pad_mode="constant")
    das = velocity.sel(offset=channels["offset"])
    for j, station in enumerate(traces):
        x = traces[station]
        y = das.sel(station=station).values
        r = np.std(x) / np.std(y)
        y = y * r
        cc = np.max(sp.correlate(x, y) / (np.std(x) * np.std(y) * len(x)))
        amplitude_ratio[i, j] = r
        corrcoef[i, j] = cc


## Plot

stations = np.array(["1", "2", "3", "4", "5", "6"])
colors = ["royalblue", "darkviolet", "lightgreen", "gold", "xkcd:pumpkin", "crimson"]

plt.rcParams["font.size"] = 8
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["lines.linewidth"] = 1

fig, ax = plt.subplots(layout="constrained", figsize=(3.3, 2.4), dpi=300)

ax.axvline(250.0, color="black", ls="--")
for station, values, color in zip(stations, corrcoef.T, colors):
    idx = np.argmax(values)
    ax.plot(wlen, values, color=color, label=station)
    ax.plot(wlen[idx], values[idx], "o", markersize=4, color=color)
ax.set_ylim(0, 1)
ax.set_ylabel("Correlation")
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.legend(loc="lower right", ncol=3)
ax.set_xlabel("Length [m]")
ax.set_xlim(0, max(wlen))

plt.savefig("figs/6_optimal_window_length_search.png")
