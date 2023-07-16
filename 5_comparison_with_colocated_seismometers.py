import cmocean.cm
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import scipy.signal as sp
import xarray as xr
import xdas.signal as xp
from matplotlib.colors import LightSource, LinearSegmentedColormap, SymLogNorm
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from obspy.signal.rotate import rotate_zne_lqt

# Parameters
freqmin = 2.5
freqmax = 15.0
gain = 1e-9
wlen = 250.0
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
fs = np.timedelta64(1, "s") / np.median(np.diff(strainrate["time"].values))
starttime = obspy.UTCDateTime(str(strainrate["time"][0].values))
endtime = obspy.UTCDateTime(str(strainrate["time"][-1].values))
strainrate["time"] = (
    strainrate["time"] - strainrate["time"][0]
).values / np.timedelta64(1, "s")

# Process DAS data
strainrate *= gain
strainrate = xp.iirfilter(strainrate, [freqmin, freqmax], "bandpass", dim="time")
deformationrate = xp.integrate(strainrate, dim="offset")
velocity_sliding = xp.sliding_mean_removal(deformationrate, wlen=wlen, dim="offset")
velocity_segment = xp.segment_mean_removal(deformationrate, limits=limits, dim="offset")

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


## PLOT

# Parameters
plt.rcParams["font.size"] = 8
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["lines.linewidth"] = 1.0
fig = plt.figure(figsize=(6.9, 8), dpi=300, layout="compressed")
gs = fig.add_gridspec(nrows=3, ncols=3)
axes = []

# Map
cmap = cmocean.cm.topo
cmap = LinearSegmentedColormap.from_list("topo", cmap(np.linspace(0.5, 1, cmap.N // 2)))

ls = LightSource(azdeg=45)
data = ls.hillshade(dtm.values, dx=5, dy=5)
hs = dtm.copy(data=data)

ax = fig.add_subplot(gs[0, 0])
axes.append(ax)
dtm.plot.imshow(
    ax=ax,
    x="x",
    y="y",
    cmap=cmap,
    vmin=0,
    vmax=1000,
    add_labels=False,
    cbar_kwargs=dict(label="Altitude [m]", orientation="horizontal"),
)
hs.plot.imshow(
    ax=ax,
    x="x",
    y="y",
    cmap="gray",
    vmin=0,
    vmax=1,
    add_labels=False,
    add_colorbar=False,
    alpha=0.3,
)
ax.plot(stations["x"], stations["y"], "v", c="black")
ax.plot(fiber["x"], fiber["y"], c="C3")
ax.plot(fiber_all["x"], fiber_all["y"], c="C3", alpha=0.5, ls=":")
for label, station in enumerate(stations["station"].values, start=1):
    ax.text(
        stations.sel(station=station)["x"],
        stations.sel(station=station)["y"] + 7.5,
        label,
        color="black",
        va="bottom",
        ha="center",
    )
for key, value in markers.items():
    ax.text(
        fiber.sel(offset=value, method="nearest")["x"],
        fiber.sel(offset=value, method="nearest")["y"] - 5,
        key,
        color="white",
        va="top",
        ha="center",
    )
ax.set_xlim(519075, 519450)
ax.set_ylim(4294175, 4294575)
ax.set_aspect("equal")
ax.add_artist(
    AnchoredSizeBar(
        ax.transData,
        100,
        "100 m",
        "lower left",
        frameon=False,
    )
)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

inset = ax.inset_axes([0.4, 0.5, 0.57, 0.47])
dtm.plot.imshow(
    ax=inset,
    x="x",
    y="y",
    cmap=cmap,
    vmin=0,
    vmax=1000,
    add_labels=False,
    add_colorbar=False,
)
hs.plot.imshow(
    ax=inset,
    x="x",
    y="y",
    cmap="gray",
    vmin=0,
    vmax=1,
    add_labels=False,
    add_colorbar=False,
    alpha=0.3,
)
inset.plot(fiber["x"], fiber["y"], c="C3")
inset.plot(fiber_all["x"], fiber_all["y"], c="C3", alpha=0.5)
inset.add_patch(
    Rectangle((519075, 4294200), 375, 400, facecolor="none", edgecolor="black")
)
inset.set_aspect("equal")
inset.xaxis.set_tick_params(labelbottom=False)
inset.yaxis.set_tick_params(labelleft=False)
inset.set_xticks([])
inset.set_yticks([])
inset.set_facecolor("xkcd:light grey blue")

# strain rate
norm = SymLogNorm(linthresh=4e-7, vmin=-4e-6, vmax=4e-6)
ax = fig.add_subplot(gs[0, 1])
axes.append(ax)
strainrate.plot.imshow(
    ax=ax,
    cmap="viridis",
    norm=norm,
    robust=True,
    add_labels=False,
    cbar_kwargs=dict(
        label=r"Strain rate [$\rm{\varepsilon.s^{-1}}$]",
        orientation="horizontal",
        extend="neither",
    ),
    yincrease=False,
)
ax.set_xlabel("Offset [m]")
ax.set_ylabel("Time [s]")

# deformation rate
norm = SymLogNorm(linthresh=4e-6, vmin=-4e-5, vmax=4e-5)
ax = fig.add_subplot(gs[0, 2], sharex=ax, sharey=ax)
axes.append(ax)
deformationrate.plot.imshow(
    ax=ax,
    cmap="viridis",
    norm=norm,
    robust=True,
    add_labels=False,
    cbar_kwargs=dict(
        label=r"Deformation rate/Velocity [m/s]",
        orientation="horizontal",
        extend="neither",
    ),
    yincrease=False,
)
ax.set_xlabel("Offset [m]")
ax.tick_params(labelleft=False)

# segment-wise
ax = fig.add_subplot(gs[1, 0], sharex=ax, sharey=ax)
axes.append(ax)
img = velocity_segment.plot.imshow(
    ax=ax,
    cmap="viridis",
    norm=norm,
    robust=True,
    add_labels=False,
    add_colorbar=False,
    yincrease=False,
)
ax.set_ylabel("Time [s]")
ax.tick_params(labelbottom=False)


# sliding-window
ax = fig.add_subplot(gs[2, 0], sharex=ax, sharey=ax)
axes.append(ax)
velocity_sliding.plot.imshow(
    ax=ax,
    cmap="viridis",
    norm=norm,
    robust=True,
    add_labels=False,
    add_colorbar=False,
    yincrease=False,
)
ax.set_xlabel("Offset [m]")
ax.set_ylabel("Time [s]")

# annotation
for ax in axes[1:]:
    for key, value in markers.items():
        ax.axvline(
            value,
            c="black",
            ls=":",
            lw=0.75,
        )
    for key, value in enumerate(channels["offset"].values, start=1):
        ax.axvline(
            value,
            c="C3",
            ls=":",
            lw=0.75,
        )
for ax in axes[1:-1]:
    for key, value in markers.items():
        ax.annotate(
            key, (value, 1.02), xycoords=(("data", "axes fraction")), ha="center"
        )
    for key, value in enumerate(channels["offset"].values, start=1):
        ax.annotate(
            key,
            (value, 1.02),
            xycoords=(("data", "axes fraction")),
            ha="center",
            color="C3",
        )
ax.set_ylim(12, 0)

# trace comparison
t = strainrate["time"].values
scale = 1.5e-4
n = len(traces)
axs = [fig.add_subplot(gs[1, 1:])]
axs.append(fig.add_subplot(gs[2, 1:], sharex=axs[0], sharey=axs[0]))
for ax, das in zip(axs, [velocity_segment, velocity_sliding]):
    das = das.sel(offset=channels["offset"])
    axes.append(ax)
    for k, station in enumerate(traces):
        x = traces[station]
        y = das.sel(station=station).values
        amplitude_ratio = np.std(x) / np.std(y)
        y = y * amplitude_ratio
        cc = np.max(sp.correlate(x, y) / (np.std(x) * np.std(y) * len(x)))
        ax.plot(t, k * scale + x, c="black")
        ax.plot(t, k * scale + y, c="C3")
        ax.text(
            0.1,
            k * scale - 0.1 * scale,
            f"CC: {cc:.2f} - R: {amplitude_ratio:.1f}",
            fontsize=7,
        )
    ax.set_ylabel("Station")
    ax.plot([], [], c="black", label="VEL")
    ax.plot([], [], c="C3", label="DAS")
    ax.legend(loc="lower right", ncols=2, fontsize=7)
    ax.vlines(
        x=0.5, ymin=(n - 0.25) * scale, ymax=(n - 0.75) * scale, colors="black", lw=1.5
    )
    ax.text(x=0.6, y=(n - 0.5) * scale, s=r"$50 \rm{\mu m/s}$", va="center")
axs[0].tick_params(labelbottom=False)
ax.set_ylim(n * scale, -scale)
ax.set_yticks(scale * np.arange(n))
ax.set_yticklabels(np.arange(1, n + 1))
ax.set_xlim(0, 12)
ax.set_xlabel("Time [s]")

# layout
for label, ax in zip("abcdefg", axes):
    ax.add_artist(
        AnchoredText(
            f"({label})",
            loc="upper left",
            frameon=False,
            prop=dict(color="black", weight="bold"),
            pad=0.3,
            borderpad=0.0,
        )
    )
labels = [
    "Strain rate",
    "Deformation rate",
    "Segment-wise",
    "Sliding-window",
    "Segment-wise",
    "Sliding-window",
]
for label, ax in zip(labels, axes[1:]):
    ax.add_artist(
        AnchoredText(
            label,
            loc="upper right",
            prop=dict(color="black"),
            pad=0.3,
            borderpad=0.3,
        )
    )

fig.savefig("figs/5_comparison_with_colocated_seismometers.png")
