import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.offsetbox import AnchoredText


def fhann(f, L):
    "Fourier transform of the Hann function."
    out = np.zeros(f.shape)
    x = L * f
    query = np.isclose(x**2, 1.0)
    out[query] = 0.5
    out[~query] = np.sinc(x[~query]) / (1 - x[~query] ** 2)
    return out


## Radiative pattern

theta = np.linspace(0, 2 * np.pi, 361)

# reference patterns for disp. and strain
r_strain_p = np.cos(theta) * np.cos(theta)
r_strain_s = np.cos(theta) * np.sin(theta)
r_disp_p = np.cos(theta)
r_disp_s = np.sin(theta)

# patterns for hann and rect windows
L = 1.0
k = 1 / L * np.array([1, 2, 4, 8, 16])
r_deform_p_hann = np.cos(theta) * (1 - fhann(np.cos(theta) * k[:, None], L))
r_deform_s_hann = np.sin(theta) * (1 - fhann(np.cos(theta) * k[:, None], L))
r_deform_p_rect = np.cos(theta) * (1 - np.sinc(np.cos(theta) * k[:, None] * L))
r_deform_s_rect = np.sin(theta) * (1 - np.sinc(np.cos(theta) * k[:, None] * L))


## Plot

# parameters
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.size"] = 8
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["lines.linewidth"] = 1.0
plt.rcParams["mathtext.fontset"] = "cm"

fig = plt.figure(figsize=(6.4, 3.6), dpi=300, layout="constrained")
spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=[1.375, 1, 1, 0.1])

n = len(k)
colors = plt.cm.cool_r(np.linspace(0, 1, n))

# apparent wavenumber response
ax = fig.add_subplot(spec[:, 0])
kL = np.linspace(0, 16, 1001)
ax.plot(kL, 1 - np.sinc(kL), label="rect")
ax.plot(kL, 1 - fhann(kL, 1.0), label="hann")
ax.legend()
ax.set_xlim(0, 16)
ax.set_ylim(0, 1.25)
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
ax.grid(which="major")
ax.grid(which="minor", ls=":")
ax.set_xlabel("$kL$")
ax.set_aspect(8)
ax.add_artist(
    AnchoredText(
        "(a)",
        loc="upper left",
        frameon=False,
        prop=dict(color="black", weight="bold"),
        pad=0.0,
        borderpad=0.1,
    )
)

# polar response
ax = fig.add_subplot(spec[0, 1], projection="polar")
ax.axvline(0, color="black")
ax.axvline(np.pi, color="black")
ax.plot(theta, np.abs(r_strain_p), color="black", ls=":")
for idx in range(n):
    ax.plot(theta, np.abs(r_deform_p_rect)[idx], color=colors[idx], alpha=0.75)
ax.plot(theta, np.abs(r_disp_p), color="black", ls="--")
ax.grid(False)
ax.set_rticks([])
ax.set_rlim(0, 1.25)
ax.set_title("P-wave (rect)")
ax.tick_params("x", pad=-2, labelsize=7)
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
ax.set_xticks(np.deg2rad([0, 45, 90, 180, 225, 270, 315]))

# polar response
ax = fig.add_subplot(spec[0, 2], projection="polar")
ax.axvline(0, color="black")
ax.axvline(np.pi, color="black")
ax.plot(theta, np.abs(r_strain_s), color="black", ls=":", label=r"$\varepsilon$")
for idx in range(n):
    ax.plot(
        theta,
        np.abs(r_deform_s_rect)[idx],
        color=colors[idx],
        label=f"{round(L*k[idx])}",
        alpha=0.75,
    )
ax.plot(theta, np.abs(r_disp_s), color="black", ls="--", label="$u$")
ax.grid(False)
ax.set_rticks([])
ax.set_rlim(0, 1.25)
ax.set_title("S-wave (rect)")
h, l = ax.get_legend_handles_labels()
ax.tick_params("x", pad=-2, labelsize=7)
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
ax.set_xticks(np.deg2rad([0, 45, 90, 180, 225, 270, 315]))

# legend
h = h[-1:] + h[:-1]
l = l[-1:] + l[:-1]
ax = fig.add_subplot(spec[:, 3])
ax.legend(h, l, loc="center")
ax.set_axis_off()

# polar response
ax = fig.add_subplot(spec[1, 1], projection="polar")
ax.axvline(0, color="black")
ax.axvline(np.pi, color="black")
ax.plot(theta, np.abs(r_strain_p), color="black", ls=":")
for idx in range(n):
    ax.plot(theta, np.abs(r_deform_p_hann)[idx], color=colors[idx], alpha=0.75)
ax.plot(theta, np.abs(r_disp_p), color="black", ls="--")
ax.grid(False)
ax.set_rticks([])
ax.set_rlim(0, 1.25)
ax.tick_params("x", pad=-2, labelsize=7)
ax.set_title("P-wave (Hann)")
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
ax.set_xticks(np.deg2rad([0, 45, 90, 180, 225, 270, 315]))

# polar response
ax = fig.add_subplot(spec[1, 2], projection="polar")
ax.axvline(0, color="black")
ax.axvline(np.pi, color="black")
ax.plot(theta, np.abs(r_strain_s), color="black", ls=":", label=r"$\varepsilon$")
for idx in range(n):
    ax.plot(
        theta,
        np.abs(r_deform_s_hann)[idx],
        color=colors[idx],
        label=f"{round(L*k[idx])}",
        alpha=0.75,
    )
ax.plot(theta, np.abs(r_disp_s), color="black", ls="--", label="$u$")
ax.grid(False)
ax.set_rticks([])
ax.set_rlim(0, 1.25)
h, l = ax.get_legend_handles_labels()
ax.tick_params("x", pad=-2, labelsize=7)
ax.set_title("S-wave (Hann)")
ax.add_artist(
    AnchoredText(
        "(e)",
        loc="upper left",
        frameon=False,
        prop=dict(color="black", weight="bold"),
        pad=0.0,
        borderpad=0.1,
    )
)
ax.set_xticks(np.deg2rad([0, 45, 90, 180, 225, 270, 315]))

fig.savefig("figs/3_das_sensitivity_to_displacement.png")
