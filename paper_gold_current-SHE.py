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

DELTAG = 1.39 * C.E_0
REORG = 4.37 * C.E_0
DEFAULT_P_H_CATSIZE = 13
DEFAULT_P_H_CONCENT = 11
NUM_PTS = 200

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


potentials = np.linspace(-2, 0, NUM_PTS)

ph_list = [10, 11, 12, 13]

current_ph_frumkin = np.zeros((len(ph_list), NUM_PTS))
current_ph_marcus = np.zeros((len(ph_list), NUM_PTS))


model = models.DoubleLayerModel(P.DEFAULT_CONC_M, P.DEFAULT_GAMMA, 2)
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

fig = plt.figure(figsize=(4, 3))
gs = GridSpec(nrows=1, ncols=2)
ax3 = fig.add_subplot(gs[0])
ax3_marcus = fig.add_subplot(gs[1])

purple = P.get_color_gradient(len(ph_list), color="purple")

for i, p_h in enumerate(ph_list):
    ax3.plot(
        potentials + C.AU_PZC_SHE_V + 59e-3 * ph_list[-1],
        current_ph_frumkin[i, :] * 1e-1,
        color=purple[i],
        label=f"{p_h:.0f}",
    )
    ax3_marcus.plot(
        potentials + C.AU_PZC_SHE_V + 59e-3 * ph_list[-1],
        current_ph_marcus[i, :] * 1e-1,
        color=purple[i],
        label=f"{p_h:.0f}",
    )


ax3.legend(loc="lower right", frameon=False, title=r"pH")

ax3.set_ylim([-3, 0])
ax3_marcus.set_ylim([-3, 0])

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
    axis.set_xlim([-0.7, -0.2])

plt.tight_layout()

plt.savefig("figures/S1-gold-SHE-pH.pdf", dpi=240)

plt.show()
