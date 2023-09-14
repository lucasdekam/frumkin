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


DEFAULT_CONCENTRATION_M = 100e-3

ph_range = np.linspace(3, 13, 100)

gamma_list = plotting.GAMMA_LIST
sol_list = []

for gamma in gamma_list:
    model = models.AqueousVariableStern(DEFAULT_CONCENTRATION_M, gamma, 2, 4, 1)
    sol = model.ph_sweep(ph_range, tol=1e-4)
    sol_list.append(sol)

fig = plt.figure(figsize=(5, 5))
gs = GridSpec(nrows=3, ncols=2)
ax_E = fig.add_subplot(gs[0, 0])
ax_eps = fig.add_subplot(gs[0, 1], sharex=ax_E)
ax_sigma = fig.add_subplot(gs[1, 0], sharex=ax_E)
ax_fsioh = fig.add_subplot(gs[1, 1], sharex=ax_E)
ax_c = fig.add_subplot(gs[2, 0], sharex=ax_E)
ax_pressure = fig.add_subplot(gs[2, 1], sharex=ax_E)

colors = plotting.get_color_gradient(len(gamma_list))

for i, gamma in enumerate(gamma_list):
    ax_E.plot(
        ph_range,
        sol_list[i]["efield"] * 1e-9,
        color=colors[i],
        label=f"{gamma:.0f}",
    )
    ax_eps.plot(
        ph_range,
        sol_list[i]["eps"],
        color=colors[i],
        label=f"{gamma:.0f}",
    )
    ax_c.plot(
        ph_range,
        sol_list[i]["cations"],
        color=colors[i],
        label=f"{gamma:.0f}",
    )
    ax_sigma.plot(
        ph_range,
        sol_list[i]["charge"] * -100,
        color=colors[i],
        label=f"{gamma:.0f}",
    )
    ax_fsioh.plot(
        ph_range,
        (1 + (sol_list[i]["charge"] / C.E_0) / C.N_SITES_SILICA),
        color=colors[i],
        label=f"{gamma:.0f}",
    )
    ax_pressure.plot(
        ph_range,
        sol_list[i]["pressure"] / 1e9,
        color=colors[i],
        label=f"{gamma:.0f}",
    )

ax_E.set_ylabel(r"$E(0)$ / V nm$^{-1}$")
ax_E.set_ylim([-1.5, 0.05])
ax_E.set_xlim([ph_range[0], ph_range[-1]])

ax_eps.set_ylim([0, 80])
ax_eps.set_ylabel(r"$\varepsilon(0)$")
ax_eps.set_yticks([0, 20, 40, 60, 80])

ax_c.set_ylim([0, 12])
ax_c.set_ylabel(r"$c_+(x_2)$ / M")
# ax_c.set_yticks([0, 2, 4, 6, 8])

ax_sigma.set_ylim([0, 30])
ax_sigma.set_ylabel(r"$-\sigma$ / $\mu$C cm$^{-2}$")

ax_fsioh.set_ylim([0.6, 1.0])
ax_fsioh.set_ylabel(r"$f_\mathrm{SiOH}$")
# ax_fsioh.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])

ax_pressure.set_ylim([0, 0.5])
ax_pressure.set_ylabel(r"$P(0)$ / $10^4$ bar")
# ax_pressure.set_yticks([0, 0.1, 0.2])

ax_E.legend(frameon=False, title=r"$\gamma_+$", ncols=2)

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
    axis.set_xlabel(r"pH")
    axis.set_xticks([3, 5, 7, 9, 11, 13])

plt.tight_layout()

plt.savefig("figures/res-gamma-sweep-silica.pdf", dpi=240)

plt.show()
