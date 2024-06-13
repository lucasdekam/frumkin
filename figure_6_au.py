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

DELTAG = 1.37 * C.E_0
DEFAULT_PH = 11
NUM_PTS = 200

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 7
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


potentials = np.linspace(-2, 0, NUM_PTS)

conc_list = [5e-3, 250e-3, 500e-3, 1000e-3]
gamma_list = P.GAMMA_LIST
ph_list = [10, 11, 12, 13]

current_conc_frumkin = np.zeros((len(conc_list), NUM_PTS))
current_gamma_frumkin = np.zeros((len(gamma_list), NUM_PTS))

for i, conc in enumerate(conc_list):
    model = models.DoubleLayerModel(conc, P.DEFAULT_GAMMA, 2)
    sol = model.potential_sweep(potentials)
    current_conc_frumkin[i, :] = kinetics.frumkin_corrected_current(
        sol,
        deltag=DELTAG,
    )

for i, gamma in enumerate(gamma_list):
    model = models.DoubleLayerModel(P.DEFAULT_CONC_M, gamma, 2)
    sol = model.potential_sweep(potentials)
    current_gamma_frumkin[i, :] = kinetics.frumkin_corrected_current(
        sol,
        deltag=DELTAG,
    )

model = models.DoubleLayerModel(P.DEFAULT_CONC_M, P.DEFAULT_GAMMA, 2)
sol = model.potential_sweep(potentials)
current_ph_frumkin = kinetics.frumkin_corrected_current(
    sol,
    deltag=DELTAG,
)

fig = plt.figure(figsize=(7.2507112558, 2))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
# plt.subplots_adjust(left=1, bottom=0, right=1, top=0, wspace=0, hspace=0)

colors = P.get_color_gradient(len(conc_list))
red = P.get_color_gradient(len(gamma_list), color="red")
purple = P.get_color_gradient(len(ph_list), color="purple")

for i, conc in enumerate(conc_list):
    ax1.plot(
        potentials + C.AU_PZC_SHE_V + 59e-3 * DEFAULT_PH,
        current_conc_frumkin[i, :] * 1e-1,
        color=colors[i],
        label=f"{conc*1e3:.0f}",
    )

for i, gamma in enumerate(gamma_list):
    ax2.plot(
        potentials + C.AU_PZC_SHE_V + 59e-3 * DEFAULT_PH,
        current_gamma_frumkin[i, :] * 1e-1,
        color=red[i],
        label=f"{gamma:.0f}",
    )

ax3.plot(
    potentials + C.AU_PZC_SHE_V,
    current_ph_frumkin * 1e-1,
    color="k",
)

ax1.legend(loc="lower right", frameon=False, title=r"$c_+^*$ / mM")
ax2.legend(loc="lower right", frameon=False, title=r"$\gamma_+$")
ax1.set_xticks([-0.6, -0.4, -0.2, 0])
ax2.set_xticks([-0.6, -0.4, -0.2, 0])

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
    axis.set_xlabel(r"$\mathsf{E}$ vs. RHE / V")
    axis.set_ylabel(r"$j$ / mA cm$^{-2}$")
    axis.set_xlim([-0.7, 0])
    axis.set_ylim([-2, 0.1])

ax3.set_ylim([-1.5, 0.1])
ax3.set_xlim([-1.4, -0.4])
ax3.set_xlabel(r"$\mathsf{E}$ vs. SHE / V")

plt.tight_layout()
plt.subplots_adjust(left=0.13, right=0.94)
plt.savefig("figures/gr6.pdf", dpi=150)

plt.show()
