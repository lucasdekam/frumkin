"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms
from matplotlib.legend_handler import HandlerTuple
from matplotlib.gridspec import GridSpec

from edl import models
from edl import constants as C
import plotting

rcParams["font.size"] = 7
rcParams["lines.linewidth"] = 0.75
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

PHI0_V = -1
gamma_list = plotting.GAMMA_LIST
cbulk_list = plotting.CONC_LIST
DEFAULT_GAMMA = 6
DEFAULT_CONC = 1e-1
ph_list = [10, 11, 12, 13]

potentials_v_pzc = np.linspace(-2, 0, 100)

# figure setup
fig_conc = plt.figure(figsize=(7.2507112558, 2))
ax_cat_cbulk = fig_conc.add_subplot(121)
ax_oh_phbulk = fig_conc.add_subplot(122)

colors1 = plotting.get_color_gradient(len(cbulk_list))
colors3 = plotting.get_color_gradient(len(ph_list), color="purple")

# cation concentration profiles
for i, conc in enumerate(cbulk_list):
    md = models.DoubleLayerModel(conc, DEFAULT_GAMMA, 2)
    solution = md.potential_sweep(potentials_v_pzc, tol=1e-4)
    ax_cat_cbulk.plot(
        potentials_v_pzc + C.AU_PZC_SHE_V,
        solution["cat_2"],
        color=colors1[i],
        label=f"{conc*1e3:.0f}",
    )

ax_cat_cbulk.set_ylim([0, 7])
ax_cat_cbulk.set_xlim([-1.5, 0.5])
ax_cat_cbulk.set_ylabel(r"$c_+(x_2)$ / M")
ax_cat_cbulk.set_xlabel(r"E vs. SHE / V")
ax_cat_cbulk.legend(frameon=False, title=r"$c_+^*$ / mM")

# oh- concentration profiles
md = models.DoubleLayerModel(DEFAULT_CONC, DEFAULT_GAMMA, 2)
solution = md.potential_sweep(potentials_v_pzc, tol=1e-4)
for i, ph in enumerate(ph_list):
    oh_bulk = 10 ** (-14 + ph)
    oh_conc = solution["ani_2"].values / DEFAULT_CONC * oh_bulk
    ax_oh_phbulk.plot(
        potentials_v_pzc + C.AU_PZC_SHE_V,
        np.log10(oh_conc),
        color=colors3[i],
        label=f"{ph:.0f}",
    )
    # print(
    #     np.gradient(
    #         np.log(oh_conc) / C.BETA / C.E_0,
    #         potentials_v_pzc,
    #     )[0],
    # )

# ax_oh_phbulk.set_yscale("log")
ax_oh_phbulk.set_xlim([-1.5, 0.5])
# ax_oh_phbulk.set_ylim([1e-25, 1])
ax_oh_phbulk.set_ylim([-25, 1])
ax_oh_phbulk.set_ylabel(r"$\log c_\mathrm{OH^-}(x_2)$ / M")
ax_oh_phbulk.set_xlabel(r"E vs. SHE / V")
ax_oh_phbulk.legend(frameon=False, title=r"pH", loc="lower right")

# subfigure labelling
labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for label, axis in zip(labels, fig_conc.axes):
    trans = mtransforms.ScaledTranslation(-25 / 72, 10 / 72, fig_conc.dpi_scale_trans)
    axis.text(
        0.0,
        1.0,
        label,
        transform=axis.transAxes + trans,
        fontsize="medium",
        va="bottom",
    )

fig_conc.tight_layout()
fig_conc.subplots_adjust(left=0.24, right=0.79, wspace=0.5)

fig_conc.savefig("figures/gr5.pdf")
plt.show()
