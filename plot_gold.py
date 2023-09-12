"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec

from PIL import Image

from edl import models
import plotting

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


potentials = np.linspace(-1, 1, 200)

conc_list = [0, 2e-3, 6e-3, 25e-3, 100e-3]
sol_cst_stern_list = []
sol_var_stern_list = []

for conc in conc_list:
    # cst_stern = edl.Aqueous(conc + 1e-3, 5, 2, 5, 2)
    # sol = cst_stern.potential_sweep(potentials, tol=1e-4, p_h=3)
    # sol_cst_stern_list.append(sol)
    var_stern = models.AqueousVariableStern(conc + 1e-3, 5, 2, 5, 1)
    sol = var_stern.potential_sweep(potentials, tol=1e-4, p_h=3)
    sol_var_stern_list.append(sol)

fig = plt.figure(figsize=(5, 5))
gs = GridSpec(nrows=2, ncols=1)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
colors = plotting.get_color_gradient(len(conc_list))

img = np.asarray(Image.open("figures/Ojha2019_capacitance.png"))
ax1.imshow(img)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlabel(r"$\phi_\mathrm{M}$ / V vs. RHE")
ax1.set_ylabel(r"$C$ / $\mu$F cm$^{-2}$")

for i, conc in enumerate(conc_list):
    ax2.plot(
        sol_var_stern_list[i]["phi0"],
        sol_var_stern_list[i]["capacity"] * 1e2,
        color=colors[i],
        label=f"{conc*1e3:.0f}",
    )

ax2.set_xlabel(r"$\phi_\mathrm{M}$ / V")
ax2.set_ylabel(r"$C$ / $\mu$F cm$^{-2}$")
ax2.set_xlim([potentials[0], potentials[-1]])
ax2.legend()
ax2.set_ylim([0, 140])

ax2.legend(frameon=False, title=r"$c_+^\mathrm{b}$ / mM")

labels = ["(a)", "(b)"]
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

plt.savefig("figures/res-cap-gold.pdf", dpi=240)

plt.show()
