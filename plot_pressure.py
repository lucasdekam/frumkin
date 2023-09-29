"""
silica
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec

from edl import models
import plotting as P

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


potentials = np.linspace(-0.5, 0.5, 200)
conc_list = P.CONC_LIST
conc_sol_list = []

for conc in conc_list:
    model = models.AqueousVariableStern(conc, P.DEFAULT_GAMMA, 2, 4, 1)
    sol = model.potential_sweep(potentials, tol=1e-4)
    conc_sol_list.append(sol)

gamma_list = P.GAMMA_LIST
gamma_sol_list = []
for gamma in gamma_list:
    model = models.AqueousVariableStern(P.DEFAULT_CONC_M, gamma, 2, 4, 1)
    sol = model.potential_sweep(potentials, tol=1e-4)
    gamma_sol_list.append(sol)

fig = plt.figure(figsize=(5, 2))
gs = GridSpec(1, 2, figure=fig)
ax_conc = fig.add_subplot(gs[0, 0])
ax_gamma = fig.add_subplot(gs[0, 1])

colors_blu = P.get_color_gradient(len(gamma_list))
colors_red = P.get_color_gradient(len(gamma_list), color="red")

for i, conc in enumerate(conc_list):
    ax_conc.plot(
        potentials,
        conc_sol_list[i]["pressure"] * 1e-5,
        color=colors_blu[i],
        label=f"{conc*1e3:.0f}",
    )

for i, gamma in enumerate(gamma_list):
    ax_gamma.plot(
        potentials,
        gamma_sol_list[i]["pressure"] * 1e-5,
        color=colors_red[i],
        label=f"{gamma:.0f}",
    )


ax_conc.set_ylabel(r"$P(0)$ / bar")
ax_conc.set_xlabel(r"$\phi_0$ / V")
ax_gamma.set_ylabel(r"$P(0)$ / bar")
ax_gamma.set_xlabel(r"$\phi_0$ / V")

ax_conc.legend(frameon=False, title=r"$c^\mathrm{0}$ / mM")
ax_gamma.legend(frameon=False, title=r"$\gamma_+$")

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
    axis.set_ylim([0, 5000])
    axis.set_xlim([potentials[0], potentials[-1]])

plt.tight_layout()

plt.savefig("figures/res-pressure.pdf")
plt.show()
