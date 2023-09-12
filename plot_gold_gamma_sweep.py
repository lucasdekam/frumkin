"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec

from edl import models
from edl import constants as C
import plotting

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


DEFAULT_CONCENTRATION_M = 10e-3

potentials = np.linspace(-2, 0, 200)

gamma_list = [2, 4, 6, 8]
sol_list = []

for gamma in gamma_list:
    model = models.AqueousVariableStern(DEFAULT_CONCENTRATION_M, gamma, 2, 4, 1)
    sol = model.potential_sweep(potentials, tol=1e-4, p_h=11)
    sol_list.append(sol)

fig = plt.figure(figsize=(5, 5))
gs = GridSpec(nrows=3, ncols=2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax4 = fig.add_subplot(gs[1, 1], sharex=ax1)
ax5 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax6 = fig.add_subplot(gs[2, 1], sharex=ax1)

colors = plotting.get_color_gradient(len(gamma_list))

for i, gamma in enumerate(gamma_list):
    ax1.plot(
        sol_list[i]["phi0"],
        sol_list[i]["efield"] * 1e-9,
        color=colors[i],
        label=f"{gamma:.0f}",
    )
    ax2.plot(
        sol_list[i]["phi0"],
        sol_list[i]["eps"],
        color=colors[i],
        label=f"{gamma:.0f}",
    )
    ax3.plot(
        sol_list[i]["phi0"],
        sol_list[i]["cations"],
        color=colors[i],
        label=f"{gamma:.0f}",
    )
    ax4.plot(
        sol_list[i]["phi0"],
        sol_list[i]["solvent"],
        color=colors[i],
        label=f"{gamma:.0f}",
    )
    ax5.plot(
        sol_list[i]["phi0"],
        sol_list[i]["entropy"] / C.C_WATER_BULK / 1e3 / C.N_A,
        color=colors[i],
        label=f"{gamma:.0f}",
    )
    ax6.plot(
        sol_list[i]["phi0"],
        sol_list[i]["pressure"] / 1e9,
        color=colors[i],
        label=f"{gamma:.0f}",
    )

ax1.set_ylabel(r"$E(0)$ / V nm$^{-1}$")
ax1.set_ylim([-8, 0.2])
ax1.set_xlim([potentials[0], potentials[-1]])
ax1.set_yticks([-8, -6, -4, -2, 0])

ax2.set_ylim([0, 80])
ax2.set_ylabel(r"$\varepsilon(0)$")
ax2.set_yticks([0, 20, 40, 60, 80])

ax3.set_ylim([0, 12])
ax3.set_ylabel(r"$c_+(x_2)$ / M")
# ax3.set_yticks([0, 2, 4, 6, 8])

ax4.set_ylim([0, 60])
ax4.set_ylabel(r"$c_\mathrm{H_2O}(x_2)$ / M")
ax4.set_yticks([0, 20, 40, 60])

ax5.set_ylim([-2, 0.1])
ax5.set_ylabel(r"$s(0)/k_\mathrm{B}n_\mathrm{w}$")

ax6.set_ylim([0, 2.5])
ax6.set_ylabel(r"$P(0)$ / $10^4$ bar")

ax1.legend(frameon=False, title=r"$\gamma_+$", ncols=2)

labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
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
    axis.set_xlabel(r"$\phi_\mathrm{M}$ / V")

plt.tight_layout()

plt.savefig("figures/res-gamma-sweep-gold.pdf", dpi=240)

plt.show()
