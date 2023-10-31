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


potentials_v_she = np.linspace(-1.5, 0.5, 100)
potentials = potentials_v_she - C.AU_PZC_SHE_V

gamma_list = plotting.GAMMA_LIST
gamma_sol_list = []

for gamma in gamma_list:
    model = models.AqueousVariableStern(plotting.DEFAULT_CONC_M, gamma, 2, 4, 1)
    sol = model.potential_sweep(potentials, tol=1e-4, p_h=11)
    gamma_sol_list.append(sol)

conc_list = plotting.CONC_LIST
conc_sol_list = []
for conc in conc_list:
    model = models.AqueousVariableStern(conc, plotting.DEFAULT_GAMMA, 2, 4, 1)
    sol = model.potential_sweep(potentials, tol=1e-4, p_h=11)
    conc_sol_list.append(sol)

fig = plt.figure(figsize=(5, 2.25))
gs = GridSpec(nrows=1, ncols=2)
ax_conc = fig.add_subplot(gs[0, 0])
ax_gamm = fig.add_subplot(gs[0, 1])

colors1 = plotting.get_color_gradient(len(conc_list))
colors2 = plotting.get_color_gradient(len(gamma_list), color="red")

for i, conc in enumerate(conc_list):
    ax_conc.plot(
        potentials_v_she,
        conc_sol_list[i]["phi0"] - conc_sol_list[i]["phi_rp"],
        color=colors1[i],
        label=f"{conc*1e3:.0f}",
    )

for i, gamma in enumerate(gamma_list):
    ax_gamm.plot(
        potentials_v_she,
        gamma_sol_list[i]["phi0"] - gamma_sol_list[i]["phi_rp"],
        color=colors2[i],
        label=f"{gamma:.0f}",
    )

ax_conc.set_ylabel(r"$\phi_0 - \phi'$ / V")
ax_conc.set_ylim([-1.6, 0])
ax_conc.legend(frameon=False, title=r"$c_+^*$ / mM")

ax_gamm.set_ylabel(r"$\phi_0 - \phi'$ / V")
ax_gamm.set_ylim([-1.6, 0])
ax_gamm.legend(frameon=False, title=r"$\gamma_+$")

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
    # axis.set_xlabel(r"$\mathsf{E} - \mathsf{E}_\mathrm{pzc}$ / V")
    axis.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
    axis.set_xlim([potentials_v_she[0], potentials_v_she[-1]])

plt.tight_layout()

plt.savefig("figures/res-driving-force.pdf")

plt.show()
