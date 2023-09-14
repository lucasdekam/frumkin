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

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

REORG = 4.15 * C.E_0

potentials_v_rhe = np.linspace(-0.7, -0.2, 100)
potentials_v_she = potentials_v_rhe - 59e-3 * P.DEFAULT_P_H

conc_list = [5e-3, 250e-3, 500e-3, 1000e-3]
ph_list = [10, 11, 12, 13]

current_conc = np.zeros((len(conc_list), len(potentials_v_rhe)))
current_gamma = np.zeros((len(P.GAMMA_LIST), len(potentials_v_rhe)))
current_ph = np.zeros((len(ph_list), len(potentials_v_rhe)))

for i, conc in enumerate(conc_list):
    model = models.AqueousVariableStern(conc, P.DEFAULT_GAMMA, 2, 4, 1)
    current_conc[i, :] = kinetics.marcus_current(
        model,
        potentials_v_she + C.AU_PZC_SHE_V,
        pzc_she=C.AU_PZC_SHE_V,
        p_h=P.DEFAULT_P_H,
        reorg=REORG,
    )

for i, gamma in enumerate(P.GAMMA_LIST):
    model = models.AqueousVariableStern(P.DEFAULT_CONC_M, gamma, 2, 4, 1)
    current_gamma[i, :] = kinetics.marcus_current(
        model,
        potentials_v_she + C.AU_PZC_SHE_V,
        pzc_she=C.AU_PZC_SHE_V,
        p_h=P.DEFAULT_P_H,
        reorg=REORG,
    )

for i, p_h in enumerate(ph_list):
    model = models.AqueousVariableStern(P.DEFAULT_CONC_M, P.DEFAULT_GAMMA, 2, 4, 1)
    current_ph[i, :] = kinetics.marcus_current(
        model,
        potentials_v_rhe - 59e-3 * p_h + C.AU_PZC_SHE_V,
        pzc_she=C.AU_PZC_SHE_V,
        p_h=p_h,
        reorg=REORG,
    )

fig = plt.figure(figsize=(5, 4))
gs = GridSpec(nrows=2, ncols=2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)

colors = P.get_color_gradient(len(conc_list))
for i, conc in enumerate(conc_list):
    ax1.plot(
        potentials_v_rhe,
        current_conc[i, :] * 1e-1,
        color=colors[i],
        label=f"{conc*1e3:.0f}",
    )

colors = P.get_color_gradient(len(P.GAMMA_LIST))
for i, gamma in enumerate(P.GAMMA_LIST):
    ax2.plot(
        potentials_v_rhe,
        current_gamma[i, :] * 1e-1,
        color=colors[i],
        label=f"{gamma:.0f}",
    )

colors = P.get_color_gradient(len(ph_list))
for i, p_h in enumerate(ph_list):
    ax3.plot(
        potentials_v_rhe,
        current_ph[i, :] * 1e-1,
        color=colors[i],
        label=f"{p_h:.0f}",
    )

ax1.legend(loc="lower right", frameon=False, ncols=1, title=r"$c_+^\mathrm{b}$ / mM")
ax2.legend(loc="lower right", frameon=False, ncols=1, title=r"$\gamma_+$")
ax3.legend(loc="lower right", frameon=False, ncols=1, title=r"pH")

ax1.set_xlim([potentials_v_rhe[0], potentials_v_rhe[-1]])
ax2.set_xlim([potentials_v_rhe[0], potentials_v_rhe[-1]])
ax3.set_xlim([potentials_v_rhe[0], potentials_v_rhe[-1]])


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
    axis.set_xlabel(r"$\phi_\mathrm{M}$ / V vs. RHE")
    axis.set_ylabel(r"$j$ / mA cm$^{-2}$")
    axis.set_xticks([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2])

plt.tight_layout()

plt.savefig("figures/res-current-gold-marcus.pdf", dpi=240)

plt.show()
