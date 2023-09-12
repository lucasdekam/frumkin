"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl import models
from edl import constants as C
import plotting

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

PHI0_V = -1


def stern(x, slope):
    """
    Potential profile in Stern layer
    """
    return PHI0_V - slope * x


potentials = np.linspace(-1, 1, 100)
concentration_range = [0.001, 0.01, 0.1, 0.5]

capa_phi_sweep = np.zeros((len(concentration_range), len(potentials)))
sigma_phi_sweep = np.zeros((len(concentration_range), len(potentials)))
eps_phi_sweep = np.zeros((len(concentration_range), len(potentials)))

x_axes = []
phi_spatial = []
efield_spatial = []
cation_spatial = []


for i, conc in enumerate(concentration_range):
    gc = models.LangevinGouyChapmanStern(conc)
    gc_phi_sweep = gc.potential_sweep(potentials, tol=1e-3)
    capa_phi_sweep[i, :] = gc_phi_sweep["capacity"]
    sigma_phi_sweep[i, :] = gc_phi_sweep["charge"]
    eps_phi_sweep[i, :] = gc_phi_sweep["eps"]

    gc_spatial = gc.spatial_profiles(phi0=PHI0_V, tol=1e-3)
    x_axes.append(gc_spatial["x"] + C.D_ADSORBATE_LAYER * 1e9)
    phi_spatial.append(gc_spatial["phi"])
    efield_spatial.append(gc_spatial["efield"])
    cation_spatial.append(gc_spatial["cations"])


fig, ax = plt.subplots(figsize=(5, 4), nrows=2, ncols=2)
colors1 = plotting.get_color_gradient(len(concentration_range))
colors2 = plotting.get_color_gradient(len(concentration_range), color="red")

# c_entries = []
# a_entries = []

for i, conc in enumerate(concentration_range):
    ax[0, 0].plot(
        x_axes[i],
        phi_spatial[i],
        label=f"{conc*1e3:.0f}",
        color=colors1[i],
    )
    x = np.linspace(0, C.D_ADSORBATE_LAYER, 10)

    ax[0, 0].plot(
        x * 1e9,
        stern(x, efield_spatial[i][0]),
        color=colors1[i],
    )

    ax[0, 1].plot(
        potentials,
        eps_phi_sweep[i, :],
        label=f"{conc*1e3:.0f}",
        color=colors1[i],
    )

    ax[1, 0].plot(
        potentials,
        sigma_phi_sweep[i, :] * 100,
        label=f"{conc*1e3:.0f} mM",
        color=colors1[i],
    )
    ax[1, 1].plot(
        potentials,
        capa_phi_sweep[i, :] * 100,
        label=f"{conc*1e3:.0f} mM",
        color=colors1[i],
    )

ax[0, 0].set_ylabel(r"$\phi$ / V")
ax[0, 1].set_ylabel(r"$\varepsilon(x_2)$")
ax[1, 0].set_ylabel(r"$\sigma$ / $\mu$C cm$^{-2}$")
ax[1, 1].set_ylabel(r"$C$ / $\mu$F cm$^{-2}$")

ax[0, 0].set_ylim([PHI0_V, 0.05])
ax[0, 1].set_ylim([0, 80])
ax[1, 0].set_ylim([-60, 60])
ax[1, 1].set_ylim([0, 150])

ax[0, 0].set_xlabel(r"$x$ / nm")
ax[0, 1].set_xlabel(r"$\phi_\mathrm{M}$ / V")
ax[1, 0].set_xlabel(r"$\phi_\mathrm{M}$ / V")
ax[1, 1].set_xlabel(r"$\phi_\mathrm{M}$ / V")

ax[0, 0].set_xlim([0, 5])
ax[0, 1].set_xlim([potentials[0], potentials[-1]])
ax[1, 0].set_xlim([potentials[0], potentials[-1]])
ax[1, 1].set_xlim([potentials[0], potentials[-1]])

ax[0, 0].set_xticks([0, 1, 2, 3, 4, 5])

ax[0, 0].legend(frameon=False, title=r"$c_0$ / mM")

labels = ["(a)", "(b)", "(c)", "(d)"]
for label, axis in zip(labels, ax.reshape(-1)):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-25 / 72, 10 / 72, fig.dpi_scale_trans)
    axis.text(
        0.0,
        1.0,
        label,
        transform=axis.transAxes + trans,
        fontsize="medium",
        va="bottom",
    )

plt.tight_layout()
plt.savefig("figures/intro-langevin-gouy-chapman-stern.pdf")

fig2, ax2 = plt.subplots()
for i, conc in enumerate(concentration_range):
    ax2.plot(
        x_axes[i],
        cation_spatial[i],
        label=f"{conc*1e3:.0f} mM",
        color=colors1[i],
    )
ax2.set_xlim([0, 5])

plt.show()
