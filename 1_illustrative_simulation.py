import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
import xdas.signal as xp
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import blended_transform_factory

from simulate import get_source, ricker, simulate, sinus

np.random.seed(8888)


## Inputs
source_plane_wave = get_source(
    "plane",
    f0=7.5,
    p=1 / 1000.0,
    amp=1e-6,
    kind="p",
    stf=ricker,
)
source_hanging = get_source(
    "plane",
    f0=2.0,
    p=1 / 1000.0,
    amp=2e-5,
    kind="p",
    stf=sinus,
)

# geometry
limits = dict(zip(["A", "B", "C", "D", "E", "F"], [0, 500, 1000, 1500, 2500, 3000]))

# starting point
points = [150.0j]

# linear segments
points.append(points[-1] + 500.0 * np.exp(1j * np.deg2rad(-30)))
points.append(points[-1] + 500.0 * np.exp(1j * np.deg2rad(+75)))

# perturbed linear segment
n = 500
d = 500.0 / n
sigma = 3.0
dev = 0
for k in range(n):
    dev += sigma * np.random.randn()
    dev *= 0.95
    points.append(points[-1] + d * np.exp(1j * np.deg2rad(-30 + dev)))

# arc
n = 3600
L = 1000.0
for k in range(round(n) + 1):
    points.append(
        points[-1] + L / n * np.exp(1j * (np.pi / 2 - np.pi / 1.5 * k / n)).conj()
    )

# linear segment with 50m of hanging cable.
points.append(points[-1] + 225.0 * np.exp(1j * np.deg2rad(90)))
points.append(points[-1] + 25.0 * np.exp(1j * np.deg2rad(60)))
points.append(points[-1] + 25.0 * np.exp(1j * np.deg2rad(120)))
points.append(points[-1] + 225.0 * np.exp(1j * np.deg2rad(90)))

# get offsets of each point and pack everything
offset = np.pad(np.cumsum(np.abs(np.diff(points))), (1, 0))
fiber = xr.DataArray(points, {"offset": offset})


## Coupling


def coupling_plane_wave(s):
    return 1.0 - (np.abs(s - 2750) < 25).astype("float")


def coupling_hanging(s):
    return 1.0 - coupling_plane_wave(s)


## Simulation

# wavefield
x = np.linspace(0, 2000, 201)
y = np.linspace(-500, 500, 101)
xg, yg = np.meshgrid(x, y)
rg = xg + 1j * yg
data = source_plane_wave(1.0, rg)
wavefield = xr.DataArray(data, {"y": y, "x": x})

# recordings
t = np.linspace(-1.0, 2.0, 1501)

displacement_plane_wave = simulate(
    points, t, source_plane_wave, ds=1.0, output="displacement", g=coupling_plane_wave
)
deformation_plane_wave = simulate(
    points, t, source_plane_wave, ds=1.0, output="deformation", g=coupling_plane_wave
)
displacement_hanging = simulate(
    points, t, source_hanging, ds=1.0, output="displacement", g=coupling_hanging
)
deformation_hanging = simulate(
    points, t, source_hanging, ds=1.0, output="deformation", g=coupling_hanging
)
displacement = displacement_plane_wave + displacement_hanging
deformation = deformation_plane_wave + deformation_hanging
strain = xp.differentiate(deformation, dim="distance")

# conversion to displacement
segment = xp.segment_mean_removal(deformation, np.array(list(limits.values())))
sliding = xp.sliding_mean_removal(deformation, wlen=500.0)

## Plot

plt.rcParams["font.size"] = 8
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["mathtext.fontset"] = "cm"

fig = plt.figure(layout="constrained", figsize=(6.9, 6.9), dpi=300)
subfigs = fig.subfigures(nrows=2, height_ratios=[1, 3], hspace=0.15)
for subfig in subfigs:
    subfig.patch.set_facecolor("none")

ax0 = subfigs[0].subplots()
wavefield.real.plot.imshow(
    ax=ax0,
    x="x",
    y="y",
    vmin=-1e-6,
    vmax=1e-6,
    add_colorbar=False,
    add_labels=False,
    yincrease=False,
)
ax0.plot(np.real(points), np.imag(points), "C3", lw=1)
ax0.annotate(
    None,
    (1350, -440),
    (1050, -440),
    arrowprops=dict(arrowstyle="-|>", color="#fde725", capstyle="butt"),
)
ax0.set_aspect("equal", "box")
ax0.set_xlabel("x [m]")
ax0.set_ylabel("y [m]")
ax0.add_artist(
    AnchoredText(
        "(a)",
        loc="upper left",
        frameon=False,
        prop=dict(color="white", weight="bold"),
        pad=0.0,
        borderpad=0.3,
    )
)
for key, value in limits.items():
    ax0.text(
        fiber.sel(offset=value, method="nearest").real + 5,
        fiber.sel(offset=value, method="nearest").imag + 5,
        key,
        color="white",
    )
ax0.add_artist(
    FancyBboxPatch(
        boxstyle="square,pad=0.0",
        xy=(-0.15, 0.375 - 0.125 / 2),
        width=0.15,
        height=0.125,
        transform=ax0.transAxes,
        color="black",
        clip_on=False,
    )
)
ax0.text(
    -0.15 / 2,
    0.375,
    "DAS",
    color="white",
    transform=ax0.transAxes,
    va="center",
    ha="center",
    weight="bold",
    fontsize=7,
)
ax0.patch.set_facecolor("none")

axs = subfigs[1].subplots(
    nrows=3, ncols=3, sharex=True, sharey=True, gridspec_kw=dict(wspace=0.3, hspace=0.1)
)

kwargs = dict(
    vmin=-1e-6, vmax=1e-6, yincrease=False, add_colorbar=False, add_labels=False
)
deformation.plot.imshow(ax=axs[0, 0], **kwargs)
(deformation - displacement).plot.imshow(ax=axs[0, 1], **kwargs)
displacement.plot.imshow(ax=axs[0, 2], **kwargs)
(deformation - segment).plot.imshow(ax=axs[1, 1], **kwargs)
segment.plot.imshow(ax=axs[1, 2], **kwargs)
(deformation - sliding).plot.imshow(ax=axs[2, 1], **kwargs)
sliding.plot.imshow(ax=axs[2, 2], **kwargs)
strain.plot.imshow(ax=axs[2, 0], **dict(kwargs, vmin=-3e-8, vmax=3e-8))

panels = ["(b)", "(c)", "(d)", None, "(e)", "(f)", "(g)", "(h)", "(i)"]
for ax, panel in zip(axs.flat, panels):
    ax.patch.set_facecolor("none")
    if panel is None:
        ax.axis("off")
    else:
        ax.add_artist(
            AnchoredText(
                panel,
                loc="upper left",
                frameon=False,
                prop=dict(color="white", weight="bold"),
                pad=0.0,
                borderpad=0.3,
            )
        )
        for s in list(limits.values()):
            ax.axvline(s, color="C3", lw=0.75, ls=":")
for ax in axs[-1, :]:
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(500.0))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1000.0))
    ax.set_xlabel("Offset [m]")
for ax in axs[:, 0]:
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.set_ylabel("Time [s]")
for ax in axs[0, :]:
    for key, value in limits.items():
        ax.annotate(
            key, (value, 1.02), xycoords=(("data", "axes fraction")), ha="center"
        )

fig.canvas.draw()
fig.set_layout_engine(None)


ax = axs[1, 0]
ax.annotate(
    "",
    xy=(0.4 - 0.05, 0.1),
    xycoords="axes fraction",
    xytext=(0.4 - 0.05, 1.1),
    textcoords="axes fraction",
    arrowprops=dict(
        arrowstyle="-|>", connectionstyle="arc3,rad=0.3", fc="k", ec="k", lw=2
    ),
)
ax.annotate(
    "",
    xy=(0.6 - 0.05, 0.1),
    xycoords="axes fraction",
    xytext=(0.6 - 0.05, 1.1),
    textcoords="axes fraction",
    arrowprops=dict(
        arrowstyle="<|-", connectionstyle="arc3,rad=-0.3", fc="k", ec="k", lw=2
    ),
)
ax.annotate(
    r"$\dfrac{\partial \delta}{\partial s}(s,t)$",
    xy=(0.25 - 0.05, 0.6),
    xycoords="axes fraction",
    va="center",
    ha="right",
    fontsize=10,
)
ax.annotate(
    r"$\int_0^s \varepsilon(s',t)ds'$",
    xy=(0.75 - 0.05, 0.6),
    xycoords="axes fraction",
    va="center",
    ha="left",
    fontsize=10,
)


ax = axs[0, 0]
ax.annotate(
    "",
    xy=(0.5, 1.35),
    xycoords="axes fraction",
    xytext=(-0.15, 0.375),
    textcoords=ax0.transAxes,
    arrowprops=dict(
        arrowstyle="-|>", connectionstyle="arc3,rad=0.3", fc="k", ec="k", lw=2
    ),
)

axs[0, 0].text(
    0.5,
    1.175,
    "Deformation",
    ha="center",
    weight="bold",
    fontsize=10,
    transform=axs[0, 0].transAxes,
    alpha=0.6,
)
axs[0, 1].text(
    0.5,
    1.175,
    "Reference",
    ha="center",
    weight="bold",
    fontsize=10,
    transform=axs[0, 1].transAxes,
    alpha=0.6,
)
axs[0, 2].text(
    0.5,
    1.175,
    "Displacement",
    ha="center",
    weight="bold",
    fontsize=10,
    transform=axs[0, 2].transAxes,
    alpha=0.6,
)
axs[-1, 0].text(
    0.5,
    1.1,
    "Strain",
    ha="center",
    weight="bold",
    fontsize=10,
    transform=axs[-1, 0].transAxes,
    alpha=0.6,
)
axs[0, 1].text(
    -0.17,
    0.5,
    "True",
    va="center",
    rotation="vertical",
    weight="bold",
    fontsize=10,
    transform=axs[0, 1].transAxes,
    c="C0",
    alpha=0.6,
)
axs[1, 1].text(
    -0.17,
    0.5,
    "Segment-wise",
    va="center",
    rotation="vertical",
    weight="bold",
    fontsize=10,
    transform=axs[1, 1].transAxes,
    c="C1",
    alpha=0.6,
)
axs[2, 1].text(
    -0.17,
    0.5,
    "Sliding-window",
    va="center",
    rotation="vertical",
    weight="bold",
    fontsize=10,
    transform=axs[2, 1].transAxes,
    c="C2",
    alpha=0.6,
)


_, top_top = fig.transFigure.inverted().transform(
    axs[0, 0].transAxes.transform((0, 1.35))
)
_, top_bottom = fig.transFigure.inverted().transform(
    axs[0, 0].transAxes.transform((0, -0.1))
)
_, bottom_top = fig.transFigure.inverted().transform(
    axs[-1, 0].transAxes.transform((0, 1.3))
)
_, bottom_bottom = fig.transFigure.inverted().transform(
    axs[-1, 0].transAxes.transform((0, -0.3))
)
left, _ = fig.transFigure.inverted().transform(
    axs[0, 1].transAxes.transform((-0.225, 0))
)
right, _ = fig.transFigure.inverted().transform(axs[0, 2].transAxes.transform((1.1, 0)))


xa, ya = fig.transFigure.inverted().transform(axs[0, 0].transAxes.transform((0.5, 1.2)))
xb, yb = fig.transFigure.inverted().transform(axs[0, 1].transAxes.transform((0.5, 1.2)))
xc, yc = fig.transFigure.inverted().transform(axs[0, 2].transAxes.transform((0.5, 1.2)))
x1, y1 = (xa + xb) / 2, (ya + yb) / 2
x2, y2 = (xb + xc) / 2, (yb + yc) / 2
fig.text(
    x1,
    y1,
    "–",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
)
fig.text(
    x2,
    y2,
    "=",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
)


fig.add_artist(
    FancyBboxPatch(
        boxstyle="round4,pad=0.0,rounding_size=0.005",
        xy=(-0.05, top_bottom),
        width=1.13,
        height=top_top - top_bottom,
        transform=blended_transform_factory(axs[0, 0].transAxes, fig.transFigure),
        fc="lightgrey",
        ec="gray",
        alpha=0.3,
    )
)
fig.add_artist(
    FancyBboxPatch(
        boxstyle="round4,pad=0.0,rounding_size=0.005",
        xy=(-0.05, bottom_bottom),
        width=1.13,
        height=top_top - bottom_bottom,
        transform=blended_transform_factory(axs[0, 1].transAxes, fig.transFigure),
        fc="lightgrey",
        ec="gray",
        alpha=0.3,
    )
)
fig.add_artist(
    FancyBboxPatch(
        boxstyle="round4,pad=0.0,rounding_size=0.005",
        xy=(-0.05, bottom_bottom),
        width=1.13,
        height=top_top - bottom_bottom,
        transform=blended_transform_factory(axs[0, 2].transAxes, fig.transFigure),
        fc="lightgrey",
        ec="gray",
        alpha=0.3,
    )
)
fig.add_artist(
    FancyBboxPatch(
        boxstyle="round4,pad=0.0,rounding_size=0.005",
        xy=(-0.05, bottom_bottom),
        width=1.13,
        height=bottom_top - bottom_bottom,
        transform=blended_transform_factory(axs[0, 0].transAxes, fig.transFigure),
        fc="lightgrey",
        ec="gray",
        alpha=0.3,
    )
)
fig.add_artist(
    FancyBboxPatch(
        boxstyle="round4,pad=0.0,rounding_size=0.005",
        xy=(left, -0.06),
        width=right - left,
        height=1.16,
        transform=blended_transform_factory(fig.transFigure, axs[0, 0].transAxes),
        fc="none",
        ec="C0",
        alpha=0.3,
    )
)
fig.add_artist(
    FancyBboxPatch(
        boxstyle="round4,pad=0.0,rounding_size=0.005",
        xy=(left, -0.06),
        width=right - left,
        height=1.12,
        transform=blended_transform_factory(fig.transFigure, axs[1, 0].transAxes),
        fc="none",
        ec="C1",
        alpha=0.3,
    )
)
fig.add_artist(
    FancyBboxPatch(
        boxstyle="round4,pad=0.0,rounding_size=0.005",
        xy=(left, -0.06),
        width=right - left,
        height=1.12,
        transform=blended_transform_factory(fig.transFigure, axs[2, 0].transAxes),
        fc="none",
        ec="C2",
        alpha=0.3,
    )
)

fig.savefig("figs/1_illustrative_simulation.png")
