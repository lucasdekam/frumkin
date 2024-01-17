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
import plotting as P
import kinetics

DELTAG = 1.05 * C.E_0
REORG = 3 * C.E_0
DEFAULT_P_H_CATSIZE = 13
DEFAULT_P_H_CONCENT = 11
NUM_PTS = 200
X2_LIST = [3.5e-10, 4.1e-10, 5.2e-10, 5.8e-10]

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


potentials = np.linspace(-2, 0, NUM_PTS)

conc_list = [5e-3, 250e-3, 500e-3, 1000e-3]
gamma_list = P.GAMMA_LIST
ph_list = [10, 11, 12, 13]

current_conc_frumkin = np.zeros((len(conc_list), NUM_PTS))
current_gamma_frumkin = np.zeros((len(gamma_list), NUM_PTS))
current_ph_frumkin = np.zeros((len(ph_list), NUM_PTS))
current_conc_marcus = np.zeros((len(conc_list), NUM_PTS))
current_gamma_marcus = np.zeros((len(gamma_list), NUM_PTS))
current_ph_marcus = np.zeros((len(ph_list), NUM_PTS))

for i, conc in enumerate(conc_list):
    model = models.ExplicitStern(conc, P.DEFAULT_GAMMA, 2, X2_LIST[2])
    sol = model.potential_sweep(potentials)
    current_conc_frumkin[i, :] = kinetics.frumkin_corrected_current(
        sol,
        deltag=DELTAG,
    )
    current_conc_marcus[i, :] = kinetics.marcus_current(
        sol,
        pzc_she=C.AU_PZC_SHE_V,
        reorg=REORG,
    )

for i, gamma in enumerate(gamma_list):
    model = models.ExplicitStern(P.DEFAULT_CONC_M, gamma, 2, X2_LIST[i])
    sol = model.potential_sweep(potentials)
    current_gamma_frumkin[i, :] = kinetics.frumkin_corrected_current(
        sol,
        deltag=DELTAG,
    )
    current_gamma_marcus[i, :] = kinetics.marcus_current(
        sol,
        pzc_she=C.AU_PZC_SHE_V,
        reorg=REORG,
    )

model = models.ExplicitStern(P.DEFAULT_CONC_M, P.DEFAULT_GAMMA, 2, X2_LIST[2])
sol = model.potential_sweep(potentials)
for i, p_h in enumerate(ph_list):
    current_ph_frumkin[i, :] = kinetics.frumkin_corrected_current(
        sol,
        deltag=DELTAG,
    )
    current_ph_marcus[i, :] = kinetics.marcus_current(
        sol,
        pzc_she=C.AU_PZC_SHE_V,
        reorg=REORG,
    )

fig = plt.figure(figsize=(6.69423, 4))
gs = GridSpec(nrows=2, ncols=3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax1_marcus = fig.add_subplot(gs[1, 0])
ax2_marcus = fig.add_subplot(gs[1, 1])
ax3_marcus = fig.add_subplot(gs[1, 2])

colors = P.get_color_gradient(len(conc_list))
red = P.get_color_gradient(len(gamma_list), color="red")
purple = P.get_color_gradient(len(ph_list), color="purple")

for i, conc in enumerate(conc_list):
    ax1.plot(
        potentials + C.AU_PZC_SHE_V + 59e-3 * DEFAULT_P_H_CONCENT,
        current_conc_frumkin[i, :] * 1e-1,
        color=colors[i],
        label=f"{conc*1e3:.0f}",
    )
    ax1_marcus.plot(
        potentials + C.AU_PZC_SHE_V + 59e-3 * DEFAULT_P_H_CONCENT,
        current_conc_marcus[i, :] * 1e-1,
        color=colors[i],
        label=f"{conc*1e3:.0f}",
    )

for i, gamma in enumerate(gamma_list):
    ax2.plot(
        potentials + C.AU_PZC_SHE_V + 59e-3 * DEFAULT_P_H_CATSIZE,
        current_gamma_frumkin[i, :] * 1e-1,
        color=red[i],
        label=f"{gamma:.0f}",
    )
    ax2_marcus.plot(
        potentials + C.AU_PZC_SHE_V + 59e-3 * DEFAULT_P_H_CATSIZE,
        current_gamma_marcus[i, :] * 1e-1,
        color=red[i],
        label=f"{gamma:.0f}",
    )

for i, p_h in enumerate(ph_list):
    ax3.plot(
        potentials + C.AU_PZC_SHE_V + 59e-3 * p_h,
        current_ph_frumkin[i, :] * 1e-1,
        color=purple[i],
        label=f"{p_h:.0f}",
    )
    ax3_marcus.plot(
        potentials + C.AU_PZC_SHE_V + 59e-3 * p_h,
        current_ph_marcus[i, :] * 1e-1,
        color=purple[i],
        label=f"{p_h:.0f}",
    )


ax1.legend(loc="lower right", frameon=False, title=r"$c^*$ / mM")
ax2.legend(loc="lower right", frameon=False, title=r"$\gamma_+$")
ax3.legend(loc="lower right", frameon=False, title=r"pH")

ax1.set_ylim([-1, 0.1])
ax1_marcus.set_ylim([-1, 0.1])
ax2.set_ylim([-1, 0.1])
ax2_marcus.set_ylim([-1, 0.1])
ax3.set_ylim([-1, 0.1])
ax3_marcus.set_ylim([-1, 0.1])

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
    axis.set_xlabel(r"$\mathsf{E}$ / V vs. RHE")
    axis.set_ylabel(r"$j$ / mA cm$^{-2}$")
    # axis.set_xticks([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2])
    axis.set_xlim([-0.7, 1])

plt.tight_layout()

plt.savefig("figures/res-current-gold_ringe.pdf", dpi=240)

plt.show()
