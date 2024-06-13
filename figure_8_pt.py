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
rcParams["font.size"] = 7
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

ALPHA = 0.5
PH_VAL = 13
DELTAG = 0.87 * C.E_0
PT_PZC_SHE_V = 0.51

potentials_v_rhe = np.linspace(-0.2, PT_PZC_SHE_V + 59e-3 * 13, 100)

conc_list = [1e-3, 10e-3, 100e-3, 1000e-3]
ph_list = [10, 11, 12, 13]

current_conc = np.zeros((len(conc_list), len(potentials_v_rhe)))
current_gamma = np.zeros((len(P.GAMMA_LIST), len(potentials_v_rhe)))
current_ph = np.zeros((len(ph_list), len(potentials_v_rhe)))

phi2_conc = np.zeros((len(conc_list), len(potentials_v_rhe)))
phi2_gamm = np.zeros((len(P.GAMMA_LIST), len(potentials_v_rhe)))

for i, conc in enumerate(conc_list):
    model = models.DoubleLayerModel(conc, P.DEFAULT_GAMMA, 2)
    sol = model.potential_sweep(potentials_v_rhe - 59e-3 * PH_VAL - PT_PZC_SHE_V)
    current_conc[i, :] = kinetics.transport_limited_current(
        sol,
        alpha=ALPHA,
        deltag=DELTAG,
    )

for i, gamma in enumerate(P.GAMMA_LIST):
    model = models.DoubleLayerModel(P.DEFAULT_CONC_M, gamma, 2)
    sol = model.potential_sweep(potentials_v_rhe - 59e-3 * PH_VAL - PT_PZC_SHE_V)
    current_gamma[i, :] = kinetics.transport_limited_current(
        sol,
        alpha=ALPHA,
        deltag=DELTAG,
    )

for i, p_h in enumerate(ph_list):
    model = models.DoubleLayerModel(P.DEFAULT_CONC_M, P.DEFAULT_GAMMA, 2)
    sol = model.potential_sweep(potentials_v_rhe - 59e-3 * p_h - PT_PZC_SHE_V)
    current_ph[i, :] = kinetics.transport_limited_current(
        sol,
        alpha=ALPHA,
        deltag=DELTAG,
    )

fig = plt.figure(figsize=(7.2507112558, 2))  # 3.50035555836, 3.9))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

red = P.get_color_gradient(len(conc_list), color="red")
blue = P.get_color_gradient(len(P.GAMMA_LIST))

for i, conc in enumerate(conc_list):
    ax1.plot(
        potentials_v_rhe,
        current_conc[i, :] * 1e-1,
        color=blue[i],
        label=f"{conc*1e3:.0f}",
    )

for i, gamma in enumerate(P.GAMMA_LIST):
    ax2.plot(
        potentials_v_rhe,
        current_gamma[i, :] * 1e-1,
        color=red[i],
        label=f"{gamma:.0f}",
    )

ax1.set_xlim([-0.2, 0.2])
ax2.set_xlim([-0.2, 0.2])
ax1.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
ax2.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
ax1.set_xlabel(r"$\mathsf{E}$ vs. RHE / V")
ax2.set_xlabel(r"$\mathsf{E}$ vs. RHE / V")
ax1.set_ylabel(r"$j$ / mA cm$^{-2}$")
ax2.set_ylabel(r"$j$ / mA cm$^{-2}$")

ax1.legend(loc="lower right", frameon=False, title=r"$c_+^*$ / mM")
ax2.legend(loc="lower right", frameon=False, title=r"$\gamma_+$")

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
plt.subplots_adjust(left=0.24, right=0.79, wspace=0.5)
plt.savefig("figures/gr8.pdf", dpi=240)

plt.show()
