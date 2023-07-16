import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.offsetbox import AnchoredText
from PIL import Image

# Compare with buletin and old estimations
catalog_das = pd.read_csv("results/das_catalog.csv", index_col="event")
catalog_info = pd.read_csv("results/info_catalog.csv", index_col="event")
catalog_info = catalog_info.loc[catalog_das.index]
magnitudes = xr.open_dataset("results/das_magnitudes.nc").to_array("event")

plt.plot(catalog_info["magnitude"], catalog_das["magnitude"] - 0.3, ".")

# %% Unpack the saved catalog and recover information about channels above snr threshold

residuals = magnitudes - magnitudes.median("offset")
mean_residuals = residuals.median("event")
std_residuals = residuals.std("event")

limit_values = [59, 357, 926]
limit_names = ["A", "B", "C"]
limits = dict(zip(limit_names, limit_values))

# %% Parameters
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 8
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["lines.markersize"] = 5
plt.rcParams["lines.markeredgewidth"] = 0.75

# %% Plot
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    sharey=False,
    sharex=False,
    gridspec_kw=dict(width_ratios=[1, 2]),
    layout="constrained",
    figsize=(6.9, 2.5),
    dpi=300,
)

# Comparison
ax = axes[0]
xerr = catalog_info["smad"]
yerr = catalog_das["smad"]
ax.plot([-4, 4], [-4, 4], linestyle="dashed", color="orangered", lw=1)
ax.legend(["1:1"], loc="lower right")
(_, caps, _) = ax.errorbar(
    catalog_info.loc[catalog_das.index]["magnitude"],
    catalog_das["magnitude"],
    xerr=xerr,
    yerr=yerr,
    fmt="go",
    ecolor="k",
    markeredgecolor="k",
    markerfacecolor="steelblue",
    elinewidth=0.8,
    capsize=1.5,
)
for cap in caps:
    cap.set_color("k")
    cap.set_markeredgewidth(ew=0.5)
ax.set_ylabel(r"DAS $\rm{M_L}$")
ax.set_xlabel(r"Catalog $\rm{M_L}$")
ax.set_ylim([-1, 4.0])
ax.set_xlim([-1, 4.0])

# Residuals
ax = axes[1]
plt.figure()
ax.axhline(0, color="orangered", lw=1, ls="--")
ax.plot(mean_residuals["offset"], mean_residuals, "royalblue", lw=1, label="mean")
ax.fill_between(
    mean_residuals["offset"],
    mean_residuals - std_residuals,
    mean_residuals + std_residuals,
    alpha=0.5,
    edgecolor="navy",
    linewidth=1,
    facecolor="palegreen",
    label="standard deviation",
)
ax.legend(loc="lower left")
for name, value in limits.items():
    ax.axvline(value, c="k", lw=1)
    ax.annotate(
        name,
        (value, 1.01),
        xycoords=("data", "axes fraction"),
        horizontalalignment="center",
        weight="bold",
    )
ax.set_ylabel(r"$\rm{M_L}$ Residual")
ax.set_xlabel("Offset [m]")
ax.set_xlim([58, 926])
ax.set_ylim([-1, 1])


img = Image.open("data/irpinia/map.png")
aspect = img.size[1] / img.size[0]
width = 0.5
height = width * aspect
axins = ax.inset_axes([1 - width, 1 - height, width, height])
axins.set_anchor("NE")
axins.imshow(img, origin="upper")
axins.get_xaxis().set_ticks([])
axins.get_yaxis().set_ticks([])

# Layout
panels = [f"({letter})" for letter in "abc"]
for ax, panel in zip([*axes.flat, axins], panels):
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

fig.savefig("figs/8_magnitude_estimation.png")
