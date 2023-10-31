"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl import models
from edl import constants as C
import plotting

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

EFF_D_WATER_M = (C.C_WATER_BULK * 1e3 * C.N_A) ** (-1 / 3)
GAMMA = 6  # (A_M / EFF_D_WATER_M) ** 3
A_M = GAMMA ** (1 / 3) * EFF_D_WATER_M

potentials = np.linspace(-1, 1, 100)
concentration_range = plotting.CONC_LIST

sol_gcs = []
sol_lpb = []
sol_bik = []
sol_gon = []

sols_list = [sol_gcs, sol_lpb, sol_bik, sol_gon]


for i, conc in enumerate(concentration_range):
    gcs = models.GouyChapmanStern(conc, A_M / 2)
    lpb = models.LangevinPoissonBoltzmann(conc, A_M / 2)
    bik = models.Borukhov(conc, A_M)
    gon = models.AqueousVariableStern(conc, GAMMA, 2, 4, 4)

    model_list = [gcs, lpb, bik, gon]

    for model, sol in zip(model_list, sols_list):
        solution = model.potential_sweep(potentials, tol=1e-4, p_h=3)
        sol.append(solution)

# fig, ax = plt.subplots(figsize=(5, 4), nrows=2, ncols=2)
fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
axes = [ax1, ax2, ax3, ax4]

colors1 = plotting.get_color_gradient(len(concentration_range))

for axis, sol in zip(axes, sols_list):
    for i, conc in enumerate(concentration_range):
        axis.plot(
            potentials,
            sol[i]["capacity"] * 100,
            label=f"{conc*1e3:.0f}",
            color=colors1[i],
        )

ax1.legend(frameon=False, title=r"$c^*$ / mM")

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

    axis.set_ylabel(r"$C$ / $\mu$F cm$^{-2}$")
    axis.set_xlabel(r"$\phi_0$ / V")
    axis.set_xlim([potentials[0], potentials[-1]])
    axis.set_ylim([0, 150])

ax1.set_ylim([0, 250])

plt.tight_layout()
plt.savefig("figures/res-cap-comparison.pdf")
plt.show()
