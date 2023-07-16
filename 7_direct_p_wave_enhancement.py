import cmocean
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
import xdas.signal as xp
from fteikpy import Eikonal2D
from matplotlib.image import imread
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# %% Geographic data
santiago = dict(latitude=-33.4733, longitude=-70.6503)
concon = dict(latitude=-32.9597, longitude=-71.5155)
laserena = dict(latitude=-29.9448, longitude=-71.2546)

dataset = xr.open_dataset("data/chile/topo.grd")
fiber = xr.open_dataset("data/chile/fiber.nc")
fiber = fiber.interp(distance=np.arange(0, fiber["distance"][-1].values, 1000.0))

data = dataset["z"].data
data = data.reshape(dataset["dimension"].values, order="F")[:, ::-1]
lon = np.arange(
    dataset["x_range"][0],
    dataset["x_range"][1] + dataset["spacing"][0] / 2,
    dataset["spacing"][0],
)
lat = np.arange(
    dataset["y_range"][0],
    dataset["y_range"][1] + dataset["spacing"][1] / 2,
    dataset["spacing"][1],
)
topo = xr.DataArray(data, {"longitude": lon, "latitude": lat})

event = fiber.sel(distance=55000)
event["longitude"] = event["longitude"] - 0.05
event["latitude"] = event["latitude"] - 0.05


# %% Eikonal simulation
img = imread("data/chile/model.png")
bedrock = img == 0.0
water = img == 1.0
sediments = np.logical_not(np.logical_or(bedrock, water))

model_p = np.zeros(img.shape)
model_p[bedrock] = 4
model_p[sediments] = 2
model_p[water] = 0

model_s = np.zeros(img.shape)
model_s[bedrock] = 4
model_s[sediments] = 0.3
model_s[water] = 0

model_b = np.zeros(img.shape)
model_b[bedrock] = 4
model_b[sediments] = 0
model_b[water] = 0

dz = dx = 0.020
z = x = dz * np.arange(201)

eik = Eikonal2D(model_p, gridsize=(dz, dx))
tt_p = eik.solve((4.0, 0.0))

eik = Eikonal2D(model_s, gridsize=(dz, dx))
tt_s = eik.solve((4.0, 0.0))

eik = Eikonal2D(model_b, gridsize=(dz, dx))
tt_b = eik.solve((4.0, 0.0))


# %% DAS data
strainrate = xr.open_dataarray("data/chile/event.nc")
strainrate = xp.iirfilter(strainrate, 5.0, "highpass", dim="time")  # microseismic noise
strain = xp.integrate(strainrate, dim="time")
deformation = xp.integrate(strain, dim="distance")
displacement = xp.sliding_mean_removal(deformation, 1000.0, dim="distance")

# %% Plot

plt.rcParams["font.size"] = 8
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.sans-serif"] = "Helvetica"

dsel = 60500


# map & illustration
fig, axes = plt.subplots(
    figsize=(7, 2), ncols=2, width_ratios=[1, 1], gridspec_kw=dict(wspace=0.35)
)

# map
ax = axes[0]
colors_undersea = cmocean.cm.deep_r(np.linspace(0, 1, 256))
colors_land = cmocean.cm.gray(np.linspace(0.2, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = mcolors.LinearSegmentedColormap.from_list("terrain_map", all_colors)
divnorm = mcolors.TwoSlopeNorm(vmin=-6.0, vcenter=0, vmax=6)
(topo / 1000).T.plot.imshow(
    ax=ax,
    cmap=terrain_map,
    norm=divnorm,
    add_labels=False,
    cbar_kwargs=dict(
        orientation="vertical",
        extend="neither",
        label="elevation [km]",
        ticks=mticker.MultipleLocator(2.0),
    ),
)
ax.plot(fiber["longitude"], fiber["latitude"], color="C3", ls=":", lw=0.75)
ax.plot(
    fiber["longitude"].sel(distance=slice(30_000, 80_000)),
    fiber["latitude"].sel(distance=slice(30_000, 80_000)),
    color="C3",
    lw=0.75,
)
ax.scatter(
    event["longitude"], event["latitude"], s=26, marker="*", fc="w", ec="k", lw=0.3
)
ax.scatter(
    santiago["longitude"], santiago["latitude"], s=8, marker="s", fc="w", ec="k", lw=0.3
)
ax.scatter(
    concon["longitude"], concon["latitude"], s=8, marker="s", fc="w", ec="k", lw=0.3
)
ax.scatter(
    laserena["longitude"], laserena["latitude"], s=8, marker="s", fc="w", ec="k", lw=0.3
)
ax.text(
    santiago["longitude"],
    santiago["latitude"] - 0.1,
    "Santiago",
    fontsize=7,
    color="w",
    ha="center",
    va="top",
)
ax.text(
    concon["longitude"] + 0.1, concon["latitude"] + 0.1, "Concón", fontsize=7, color="w"
)
ax.text(
    laserena["longitude"] + 0.1,
    laserena["latitude"] + 0.1,
    "La Serena",
    fontsize=7,
    color="w",
)
ax.set_aspect("equal", "box")
ax.set_ylim(-35, -29)
ax.set_xlim(-74, -69)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2.0))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(1.0))
ax.yaxis.set_major_formatter(lambda x, pos: f"{abs(x):.0f}°S")
ax.yaxis.set_major_locator(mticker.MultipleLocator(2.0))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(1.0))
ax.xaxis.set_major_formatter(lambda x, pos: f"{abs(x):.0f}°W")
ax.add_artist(
    AnchoredText(
        "(a)",
        loc="upper left",
        frameon=False,
        prop=dict(color="white", weight="bold"),
        pad=0.0,
        borderpad=0.1,
    )
)

# eikonal simulation
ax = axes[1]
ax.pcolormesh(z, x, img, cmap="gray", vmin=-1, rasterized=True)
ax.contour(z, x, tt_s.grid, colors="C1", levels=np.arange(0, 5, 0.30), linewidths=0.5)
ax.contour(z, x, tt_p.grid, colors="C0", levels=np.arange(0, 5, 0.30), linewidths=0.5)
ax.contour(z, x, tt_b.grid, colors="k", levels=np.arange(0, 5, 0.30), linewidths=0.5)
ax.contour(z, x, tt_p.grid, colors="k", levels=[5], linewidths=0.5)
ax.contour(z, x, tt_b.grid, colors="k", levels=[5], linewidths=0.5)
ax.set_xlabel("x [km]")
ax.set_ylabel("z [km]")
ax.set_ylim(0, 2)
ax.set_aspect("equal", "box")
ax.invert_yaxis()
ax.add_artist(
    AnchoredText(
        "(b)",
        loc="upper left",
        frameon=False,
        prop=dict(color="black", weight="bold"),
        pad=0.0,
        borderpad=0.1,
    )
)
ax.text(0.7, 1.65, "X", fontdict=dict(weight="bold"))
ax.text(1.3, 0.8, "Xp", fontdict=dict(weight="bold"), color="C0")
ax.text(2.0, 0.4, "Xs", fontdict=dict(weight="bold"), color="C1")

fig.savefig("figs/7ab_direct_p_wave_enhancement.png", bbox_inches="tight")


# wavefields
fig, axes = plt.subplots(
    figsize=(7, 5),
    nrows=2,
    ncols=2,
    sharey=True,
    gridspec_kw=dict(width_ratios=[1, 10], wspace=0, hspace=0),
)

# strain rate trace
ax = axes[0, 0]
tr = strainrate.sel(distance=dsel, method="nearest")
ax.plot(1e6 * tr, tr["time"], lw=0.5, c="k")
ax.set_ylabel("Time [s]")
ax.set_xlim(1, -1)
ax.add_artist(
    AnchoredText(
        "(c)",
        loc="upper left",
        frameon=False,
        prop=dict(color="black", weight="bold"),
        pad=0.0,
        borderpad=0.1,
    )
)
ax.add_artist(
    AnchoredSizeBar(
        ax.transData,
        1.0,
        r"1  $\mu\varepsilon /s$",
        loc="lower center",
        frameon=False,
        color="gray",
        fontproperties=dict(size=7),
    )
)
ax.tick_params(bottom=False, labelbottom=False)

# strain rate wavefield
ax = axes[0, 1]
strainrate[::2, ::2].plot.imshow(
    ax=ax,
    add_colorbar=True,
    add_labels=False,
    norm=mcolors.SymLogNorm(1e-8, vmin=-1e-6, vmax=1e-6),
    cmap="viridis",
    interpolation="none",
    cbar_kwargs=dict(pad=0, extend="neither", label=r"Strain rate [$\varepsilon$/s]"),
)
ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
ax.axvline(dsel, c="C3", ls="-", lw=0.5)
ax.set_xlim(30_000, 80_000)

# deformation/displacement trace
ax = axes[1, 0]
tr = displacement.sel(distance=dsel, method="nearest")
ax.plot(1e6 * tr, tr["time"], lw=0.5, c="k")
ax.set_xlim(-0.5, 0.5)
ax.set_ylabel("Time [s]")
axins = ax.inset_axes([0.6, 0.65, 0.4, 0.35])
axins.plot(tr, tr["time"], lw=0.5, c="k")
axins.set_xlim(-0.1, 0.1)
axins.set_ylim(3.5, 0)
axins.set_xlim(-3e-8, 3e-8)
axins.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
ax.add_artist(
    AnchoredText(
        "(d)",
        loc="upper left",
        frameon=False,
        prop=dict(color="black", weight="bold"),
        pad=0.0,
        borderpad=0.1,
    )
)
ax.add_artist(
    AnchoredSizeBar(
        ax.transData,
        0.5,
        r"0.5  $\mu m$",
        loc="lower center",
        frameon=False,
        color="gray",
        fontproperties=dict(size=7),
    )
)
ax.tick_params(bottom=False, labelbottom=False)

# displacement wavefield
ax = axes[1, 1]
displacement[::2, ::2].plot.imshow(
    ax=ax,
    add_colorbar=True,
    add_labels=False,
    norm=mcolors.SymLogNorm(1e-9, vmin=-1e-6, vmax=1e-6),
    cmap="viridis",
    interpolation="none",
    cbar_kwargs=dict(pad=0, extend="neither", label="Displacement [m]"),
)
ax.axvline(dsel, c="C3", ls="-", lw=0.5)
ax.text(75000, 7.7, "Ss", color="C3", fontdict=dict(weight="bold"))
ax.text(72500, 3.8, "Ps", color="C3", fontdict=dict(weight="bold"))
ax.text(69000, 2.5, "Pp", color="C3", fontdict=dict(weight="bold"))


ax.set_xlim(30_000, 80_000)
axins.tick_params(left=False, labelbottom=False)
ax.tick_params(left=False, labelleft=False)
ax.set_xlabel("Offset [m]")
ax.set_ylim(10, 0)

fig.savefig("figs/7cd_direct_p_wave_enhancement.png", bbox_inches="tight")
