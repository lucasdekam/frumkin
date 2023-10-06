"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl import models
import plotting

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

PHI0_V = -1

potentials = np.linspace(-1, 1, 100)
gamma_range = plotting.GAMMA_LIST

sols = []
sweeps = []

for i, gamma in enumerate(gamma_range):
    model = models.AqueousVariableStern(plotting.DEFAULT_CONC_M, gamma, 2, 5, 1)
    sweep = model.potential_sweep(potentials, tol=1e-3)
    sweeps.append(sweep)

    spatial = model.spatial_profiles(phi0=PHI0_V, tol=1e-3)
    spatial["x_shifted"] = spatial["x"] + model.get_stern_layer_thickness(PHI0_V) * 1e9
    sols.append(spatial)

# fig, ax = plt.subplots(figsize=(5, 4), nrows=2, ncols=2)
fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

colors1 = plotting.get_color_gradient(len(gamma_range), color="red")
# colors2 = plotting.get_color_gradient(len(gamma_range), color="red")

# c_entries = []
# a_entries = []

for i, gamma in enumerate(gamma_range):
    ax1.plot(
        sols[i]["x"],
        sols[i]["phi"],
        label=f"{gamma:.0f}",
        color=colors1[i],
    )

    ax2.plot(
        sols[i]["x"],
        sols[i]["cations"],
        label=f"{gamma:.0f}",
        color=colors1[i],
    )

    ax3.plot(
        sols[i]["x"],
        sols[i]["eps"],
        label=f"{gamma:.0f}",
        color=colors1[i],
    )
    ax4.plot(
        potentials,
        sweeps[i]["capacity"] * 100,
        label=f"{gamma:.0f}",
        color=colors1[i],
    )

ax1.set_ylabel(r"$\phi$ / V")
ax2.set_ylabel(r"$c_+$ / M")
ax3.set_ylabel(r"$\varepsilon/\varepsilon_0$")
ax4.set_ylabel(r"$C$ / $\mu$F cm$^{-2}$")

ax1.set_ylim([PHI0_V, 0.05])
ax2.set_ylim([0, 8])
ax3.set_ylim([0, 80])
ax4.set_ylim([0, 150])

ax1.set_xlabel(r"$x$ / nm")
ax2.set_xlabel(r"$x$ / nm")
ax3.set_xlabel(r"$x$ / nm")
ax4.set_xlabel(r"$\phi_0$ / V")

ax1.set_xlim([0, 1])
ax2.set_xlim([0, 5])
ax3.set_xlim([0, 5])
ax4.set_xlim([potentials[0], potentials[-1]])

# ax[0, 0].set_xticks([0, 1, 2, 3, 4, 5])

ax1.legend(frameon=False, title=r"$\gamma_+$")

labels = ["(a)", "(b)", "(c)", "(d)"]
for label, axis in zip(labels, fig.axes):
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
plt.savefig("figures/intro-gongadze-iglic-gamma.pdf")
plt.show()
