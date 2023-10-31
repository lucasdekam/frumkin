"""
silica
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl import models
from edl import constants as C
import plotting as P

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


def entropy(bpe):
    """
    Calculate the volumetric entropy density s/kb
    """
    s_over_nkb = 1 - bpe / np.tanh(bpe) + np.log(np.sinh(bpe) / bpe)
    return s_over_nkb


potentials = np.linspace(-1, 1, 200)
conc_list = P.CONC_LIST
conc_sol_list = []

for conc in conc_list:
    model = models.AqueousVariableStern(conc, P.DEFAULT_GAMMA, 2, 4, 4)
    sol = model.potential_sweep(potentials, tol=1e-4)
    conc_sol_list.append(sol)

gamma_list = P.GAMMA_LIST
gamma_sol_list = []
for gamma in gamma_list:
    model = models.AqueousVariableStern(P.DEFAULT_CONC_M, gamma, 2, 4, 4)
    sol = model.potential_sweep(potentials, tol=1e-4)
    gamma_sol_list.append(sol)

fig = plt.figure(figsize=(5, 4))
# gs = GridSpec(2, 2, figure=fig)
ax_entropy = fig.add_subplot(221)
ax_conc = fig.add_subplot(222)
ax_gamma = fig.add_subplot(223)

bpe = np.linspace(-4, 4, 100)
ax_entropy.plot(bpe, entropy(bpe), "k")

colors_blu = P.get_color_gradient(len(gamma_list))
colors_red = P.get_color_gradient(len(gamma_list), color="red")

for i, conc in enumerate(conc_list):
    ax_conc.plot(
        conc_sol_list[i]["charge"] * 100,
        conc_sol_list[i]["entropy"] / (C.C_WATER_BULK * 1e3 * C.N_A),
        color=colors_blu[i],
        label=f"{conc*1e3:.0f}",
    )

for i, gamma in enumerate(gamma_list):
    ax_gamma.plot(
        gamma_sol_list[i]["charge"] * 100,
        gamma_sol_list[i]["entropy"] / (C.C_WATER_BULK * 1e3 * C.N_A),
        color=colors_red[i],
        label=f"{gamma:.0f}",
    )


ax_conc.legend(frameon=False, title=r"$c^*$ / mM")
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
    # axis.set_ylim([0, 15000])
    axis.set_xlim([-10, 10])
    axis.set_ylim([-0.15, 0.01])
    axis.set_ylabel(r"$s(0)a^3/k_\mathrm{B}$")
    axis.set_xlabel(r"$\sigma$ / $\mu$C cm$^{-2}$")

ax_entropy.set_ylabel(r"$s/n_\mathrm{w}k_\mathrm{B}$")
ax_entropy.set_xlabel(r"$\beta p E$")
ax_entropy.set_xlim([bpe[0], bpe[-1]])
ax_entropy.set_ylim([-1.5, 0.05])
plt.tight_layout()

plt.savefig("figures/res-entropy.pdf")
plt.show()
