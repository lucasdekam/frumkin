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

ALPHA = 0.5
PH_CONC = 11
PH_GAMMA = 13

potentials_v_rhe = np.linspace(-0.2, C.PT_PZC_SHE_V + 59e-3 * 13, 100)

conc_list = [1e-3, 10e-3, 100e-3, 1000e-3]
ph_list = [10, 11, 12, 13]

current_conc = np.zeros((len(conc_list), len(potentials_v_rhe)))
current_gamma = np.zeros((len(P.GAMMA_LIST), len(potentials_v_rhe)))
current_ph = np.zeros((len(ph_list), len(potentials_v_rhe)))

phi2_conc = np.zeros((len(conc_list), len(potentials_v_rhe)))
phi2_gamm = np.zeros((len(P.GAMMA_LIST), len(potentials_v_rhe)))

for i, conc in enumerate(conc_list):
    model = models.AqueousVariableStern(conc, P.DEFAULT_GAMMA, 2, 4, 1)
    sol = model.potential_sweep(
        potentials_v_rhe - 59e-3 * PH_CONC - C.PT_PZC_SHE_V, p_h=PH_CONC
    )
    current_conc[i, :] = kinetics.transport_limited_current(
        sol,
        alpha=ALPHA,
    )
    phi2_conc[i, :] = sol["phi_rp"].values

for i, gamma in enumerate(P.GAMMA_LIST):
    model = models.AqueousVariableStern(P.DEFAULT_CONC_M, gamma, 2, 4, 1)
    sol = model.potential_sweep(
        potentials_v_rhe - 59e-3 * PH_GAMMA - C.PT_PZC_SHE_V, p_h=PH_CONC
    )
    current_gamma[i, :] = kinetics.transport_limited_current(
        sol,
        alpha=ALPHA,
    )
    phi2_gamm[i, :] = sol["phi_rp"].values

for i, p_h in enumerate(ph_list):
    model = models.AqueousVariableStern(P.DEFAULT_CONC_M, P.DEFAULT_GAMMA, 2, 4, 1)
    sol = model.potential_sweep(
        potentials_v_rhe - 59e-3 * p_h - C.PT_PZC_SHE_V, p_h=PH_CONC
    )
    current_ph[i, :] = kinetics.transport_limited_current(
        sol,
        alpha=ALPHA,
    )

fig = plt.figure(figsize=(5, 6))
gs = GridSpec(nrows=3, ncols=2)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[2, 0])

blue = P.get_color_gradient(len(conc_list))
red = P.get_color_gradient(len(conc_list), color="red")
purple = P.get_color_gradient(len(conc_list), color="purple")

for i, conc in enumerate(conc_list):
    axa.plot(
        potentials_v_rhe - 59e-3 * PH_CONC,
        phi2_conc[i, :],
        color=blue[i],
        label=f"{conc*1e3:.0f}",
    )
    ax1.plot(
        potentials_v_rhe,
        current_conc[i, :] * 1e-1,
        color=blue[i],
        label=f"{conc*1e3:.0f}",
    )

blue = P.get_color_gradient(len(P.GAMMA_LIST))
for i, gamma in enumerate(P.GAMMA_LIST):
    axb.plot(
        potentials_v_rhe - 59e-3 * PH_GAMMA,
        phi2_gamm[i, :],
        color=red[i],
        label=f"{gamma:.0f}",
    )
    ax2.plot(
        potentials_v_rhe,
        current_gamma[i, :] * 1e-1,
        color=red[i],
        label=f"{gamma:.0f}",
    )

blue = P.get_color_gradient(len(ph_list))
for i, p_h in enumerate(ph_list):
    ax3.plot(
        potentials_v_rhe,
        current_ph[i, :] * 1e-1,
        color=purple[i],
        label=f"{p_h:.0f}",
    )

# axa.set_xlim([-0.2])
axa.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
axb.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
axa.set_xlim([-0.8, C.PT_PZC_SHE_V])
axb.set_xlim([-0.8, C.PT_PZC_SHE_V])
axa.set_ylabel(r"$\phi'$ / V")
axb.set_ylabel(r"$\phi'$ / V")
axa.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
axb.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
axa.set_ylim([-0.8, 0.0])
axb.set_ylim([-0.8, 0.0])

ax1.set_xlim([-0.2, 0.1])
ax2.set_xlim([-0.2, 0.1])
ax3.set_xlim([-0.2, 0.1])
ax1.set_xlabel(r"$\mathsf{E}$ / V vs. RHE")
ax2.set_xlabel(r"$\mathsf{E}$ / V vs. RHE")
ax3.set_xlabel(r"$\mathsf{E}$ / V vs. RHE")
ax1.set_ylabel(r"$j$ / mA cm$^{-2}$")
ax2.set_ylabel(r"$j$ / mA cm$^{-2}$")
ax3.set_ylabel(r"$j$ / mA cm$^{-2}$")

axa.legend(loc="lower right", frameon=False, title=r"$c_0$ / mM")
axb.legend(loc="lower right", frameon=False, title=r"$\gamma_+$")
ax3.legend(loc="lower right", frameon=False, title=r"pH")

labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
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

plt.savefig("figures/res-current-pt.pdf", dpi=240)

plt.show()
