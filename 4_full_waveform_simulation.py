import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xdas.signal as xp
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def cov(x, y=None, dim="time"):
    if y is None:
        y = x
    return ((x - x.mean(dim)) * (y - y.mean(dim))).mean(dim)


# load simulation
velocity = xr.open_dataarray("data/irpinia/simulation.nc")
limits = velocity["offset"][[0, -1]].values
length = limits[1] - limits[0]

## Conversions

# velocity to strain rate
strain_rate = xp.differentiate(velocity, midpoints=True, dim="offset")

# velocity to deformation
deformation = velocity - velocity[:, 0]

# deformation to velocity
recovered_segment = xp.segment_mean_removal(deformation, limits, dim="offset")
recovered_sliding = xp.sliding_mean_removal(deformation, length, dim="offset")

## Error quantification
mse_segment = np.square(recovered_segment - velocity).mean("time")
pmse_segment = mse_segment / np.square(velocity).mean("time")
corr_segment = cov(recovered_segment, velocity) / np.sqrt(
    cov(recovered_segment) * cov(velocity)
)
mse_sliding = np.square(recovered_sliding - velocity).mean("time")
pmse_sliding = mse_sliding / np.square(velocity).mean("time")
corr_sliding = cov(recovered_sliding, velocity) / np.sqrt(
    cov(recovered_sliding) * cov(velocity)
)


##  Plots

# parameters
plt.rcParams["font.size"] = 8
plt.rcParams["font.sans-serif"] = "Helvetica"
linthresh = 1 / 10

fig = plt.figure(
    figsize=(6.9, 4.8),
    dpi=300,
)
gs = fig.add_gridspec(
    nrows=3,
    ncols=4,
    left=0.08,
    bottom=0.19,
    right=0.96,
    top=0.97,
    wspace=0.1,
    hspace=0.5,
    width_ratios=None,
    height_ratios=[2, 0.075, 0.75],
)

axes = []
for k in range(4):
    if k == 0:
        axes.append(fig.add_subplot(gs[0, 0]))
    else:
        axes.append(fig.add_subplot(gs[0, k], sharex=axes[0], sharey=axes[0]))
cax = fig.add_subplot(gs[1, 1:3])

# wavefields
wavefields = [velocity, strain_rate, recovered_segment, recovered_sliding]
for ax, xarr in zip(axes, wavefields):
    img = (xarr / np.max(np.abs(xarr.values))).plot.imshow(
        ax=ax,
        norm=colors.SymLogNorm(linthresh=linthresh, vmin=-1, vmax=1),
        cmap="viridis",
        add_labels=False,
        add_colorbar=False,
    )
fig.colorbar(img, cax=cax, ax=axes, orientation="horizontal")
for ax in axes:
    ax.set_xlim([velocity["offset"][0], velocity["offset"][-1]])
    ax.set_ylim(5, 0)
    ax.tick_params(labelleft=False)
    ax.set_xlabel(r"Offset $[\rm{m}]$")
axes[0].tick_params(labelleft=True)
axes[0].set_ylabel(r"Time $[\rm{s}]$")
panels = [f"({letter})" for letter in "abcd"]
labels = ["Velocity", "Strain rate", "Segment-wise", "Sliding-window"]
for ax, panel, label in zip(axes, panels, labels):
    ax.add_artist(
        AnchoredText(
            panel,
            loc="upper left",
            frameon=False,
            prop=dict(color="black", weight="bold"),
            pad=0.3,
            borderpad=0.0,
        )
    )
    ax.add_artist(
        AnchoredText(
            label,
            loc="upper right",
            prop=dict(color="black"),
            pad=0.3,
            borderpad=0.3,
        )
    )


# errors
ax = fig.add_subplot(gs[2, 0:2])
ax.plot(corr_segment["offset"], corr_segment, ls="--", c="C0", lw=1)
ax.plot(mse_segment["offset"], mse_segment, ls="--", c="C2", lw=1)
ax.plot(mse_segment["offset"], pmse_segment, ls="--", c="C1", lw=1)
ax.plot(corr_sliding["offset"], corr_sliding, ls="-", c="C0", lw=1)
ax.plot(mse_sliding["offset"], mse_sliding, ls="-", c="C2", lw=1)
ax.plot(mse_sliding["offset"], pmse_sliding, ls="-", c="C1", lw=1)
ax.plot([], [], ls="-", c="C0", label="Corr")
ax.plot([], [], ls="-", c="C2", label="MSE")
ax.plot([], [], ls="-", c="C1", label="PMSE")
ax.plot([], [], ls="-", c="white", label=" ")
ax.plot([], [], ls="--", c="black", label="Segment")
ax.plot([], [], ls="-", c="black", label="Sliding")
ax.legend(
    loc="lower left",
    ncols=3,
    bbox_to_anchor=(0.1, -1.1, 0.8, 0.2),
    borderaxespad=0.0,
    mode="expand",
)
ax.set_xlim([velocity["offset"][0], velocity["offset"][-1]])
ax.set_ylim(0, 1)
ax.set_xlabel(r"Offset $[\rm{m}]$")
ax.add_artist(
    AnchoredText(
        "(e)",
        loc="upper left",
        frameon=False,
        prop=dict(color="black", weight="bold"),
        pad=0.3,
        borderpad=0.0,
    )
)

ax = fig.add_subplot(gs[2, 2:4])

p1 = [-240, 0]
p2 = [240, 0]
p3 = [-225, -7]
p4 = [225, -7]
p5 = [-185, -21]
p6 = [185, -21]
p7 = [-250, -0]
p8 = [250, -0]
p9 = [250, -60]
p10 = [-250, -60]
p11 = [-120, -25]
p12 = [120, -25]
p13 = [-180, -10]
p14 = [180, -10]
p15 = [0, 0]
p16 = [0, -10]

layers = dict(
    left=Polygon(
        [p15, p1, p3, p13, p16, p15],
        fc="xkcd:beige",
        ec="black",
        lw=0.5,
        label=r"$\rm{c_P}$ = 500 m/s - $\rm{c_S}$ = 60 m/s - ρ = 1500 kg/m³",
    ),
    right=Polygon(
        [p2, p15, p16, p14, p4],
        fc="xkcd:pinkish grey",
        ec="black",
        lw=0.5,
        label=r"$\rm{c_P}$ = 2000 m/s - $\rm{c_S}$ = 300 m/s - ρ = 2000 kg/m³",
    ),
    bottom=Polygon(
        [p3, p5, p11, p12, p6, p4, p14, p16, p13],
        fc="xkcd:light blue gray",
        ec="black",
        lw=0.5,
        label=r"$\rm{c_P}$ = 1500 m/s - $\rm{c_S}$ = 130 m/s - ρ = 2000 kg/m³",
    ),
    bedrock=Polygon(
        [p1, p7, p10, p9, p8, p2, p4, p6, p12, p11, p5, p3],
        fc="xkcd:silver",
        ec="black",
        lw=0.5,
        label=r"$\rm{c_P}$ = 3250 m/s - $\rm{c_S}$ = 1730 m/s - ρ = 2720 kg/m³",
    ),
)
for key in layers:
    ax.add_patch(layers[key])
ax.plot([-175, 175], [0, 0], c="C3", lw=1, label="Fiber-optic cable")
ax.set_xlim(-250, 250)
ax.set_ylim(-60, 40)
ax.set_aspect("equal", anchor="N")
ax.legend(
    bbox_to_anchor=(0.09, -0.45, 0.82, 0.2),
    borderaxespad=0.0,
    mode="expand",
    fontsize=7,
)
ax.add_artist(
    AnchoredSizeBar(
        ax.transData,
        50,
        "50 m",
        loc="lower left",
        frameon=False,
        color="black",
    )
)
ax.add_artist(
    AnchoredText(
        "(f)",
        loc="upper left",
        frameon=False,
        prop=dict(color="black", weight="bold"),
        pad=0.3,
        borderpad=0.0,
    )
)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

fig.align_labels()
fig.savefig("figs/4_full_waveform_simulation.png")

# # outputs
# print(f"Median segment MSE: {np.median(mse_segment):.3f}")
# print(f"Median segment PMSE: {np.median(pmse_segment * 100):.1f}%")
# print(f"Median segment Correlation: {np.median(corr_segment * 100):.1f}%")
# print(f"Median sliding MSE: {np.median(mse_sliding):.3f}")
# print(f"Median sliding PMSE: {np.median(pmse_sliding * 100):.1f}%")
# print(f"Median sliding Correlation: {np.median(corr_sliding * 100):.1f}%")
# print(f"Mean segment MSE: {np.mean(mse_segment):.3f}")
# print(f"Mean segment PMSE: {np.mean(pmse_segment * 100):.1f}%")
# print(f"Mean segment Correlation: {np.mean(corr_segment * 100):.1f}%")
# print(f"Mean sliding MSE: {np.mean(mse_sliding):.3f}")
# print(f"Mean sliding PMSE: {np.mean(pmse_sliding * 100):.1f}%")
# print(f"Mean sliding Correlation: {np.mean(corr_sliding * 100):.1f}%")
