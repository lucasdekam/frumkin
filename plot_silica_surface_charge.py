"""
silica
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec

from edl import models
from edl import constants as C
import plotting as P

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


DEFAULT_CONC_M = 10e-3

ph_range = np.linspace(3, 13, 100)

conc_list = P.CONC_LIST
conc_sol_list = []

for conc in conc_list:
    model = models.AqueousVariableStern(conc, P.DEFAULT_GAMMA, 2, 4, 1)
    sol = model.ph_sweep(ph_range, tol=1e-4)
    conc_sol_list.append(sol)

gamma_list = P.GAMMA_LIST
gamma_sol_list = []
for gamma in gamma_list:
    model = models.AqueousVariableStern(DEFAULT_CONC_M, gamma, 2, 4, 1)
    sol = model.ph_sweep(ph_range, tol=1e-4)
    gamma_sol_list.append(sol)

gamm6 = models.AqueousVariableStern(DEFAULT_CONC_M, P.DEFAULT_GAMMA, 2, 4, 1)
solgamm6 = gamm6.ph_sweep(ph_range, tol=1e-4)

gc = models.AqueousVariableStern(DEFAULT_CONC_M, 0, 0, 0, 0, eps_r_opt=C.EPS_R_WATER)
solgc = gc.ph_sweep(ph_range, tol=1e-4)

# lpb = models.LPBMultispecies(DEFAULT_CONC_M)
# sollpb = lpb.ph_sweep(ph_range, tol=1e-4)
# print(
#     sollpb["solvent"].values[-1], sollpb["cations"].values[-1], sollpb["eps"].values[-1]
# )

p0 = models.AqueousVariableStern(
    DEFAULT_CONC_M, P.DEFAULT_GAMMA, 2, 4, 1, eps_r_opt=C.EPS_R_WATER
)
solp0 = p0.ph_sweep(ph_range, tol=1e-4)

fig = plt.figure(figsize=(5, 4))
# gs = GridSpec(2, 8, figure=fig, height_ratios=[2, 2])
ax_models = fig.add_subplot(221)
ax_sigma_conc = fig.add_subplot(222)
ax_sigma_gamma = fig.add_subplot(223)

colors_blu = P.get_color_gradient(len(gamma_list))
colors_red = P.get_color_gradient(len(gamma_list), color="red")

for i, conc in enumerate(conc_list):
    ax_sigma_conc.plot(
        ph_range,
        -conc_sol_list[i]["charge"] * 100,
        color=colors_blu[i],
        label=f"{conc*1e3:.0f}",
    )


ax_models.plot(ph_range, -solgc["charge"] * 100, "-", color="black", label="GCS")
# ax_models.plot(ph_range, -sollpb["charge"] * 100, ":", color="black", label="LPB")
ax_models.plot(ph_range, -solp0["charge"] * 100, "--", color="black", label="GI (p=0)")
ax_models.plot(ph_range, -solgamm6["charge"] * 100, "-.", color="black", label="GI")

for i, gamma in enumerate(gamma_list):
    ax_sigma_gamma.plot(
        ph_range,
        -gamma_sol_list[i]["charge"] * 100,
        color=colors_red[i],
        label=f"{gamma:.0f}",
    )


ax_models.set_ylim([0, 80])
ax_models.set_ylabel(r"$-\sigma$ / $\mu$C cm$^{-2}$")
ax_models.set_xlim([3, 13])
ax_models.set_xlabel("pH")
ax_sigma_conc.set_ylabel(r"$-\sigma$ / $\mu$C cm$^{-2}$")
ax_sigma_gamma.set_ylabel(r"$-\sigma$ / $\mu$C cm$^{-2}$")
ax_sigma_conc.set_ylim([0, 25])
ax_sigma_gamma.set_ylim([0, 25])

ax_models.legend(frameon=False)
ax_sigma_conc.legend(frameon=False, title=r"$c_+^*$ / mM")
ax_sigma_gamma.legend(frameon=False, title=r"$\gamma_+$")

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
    axis.set_xlabel(r"pH")
    axis.set_xticks([3, 5, 7, 9, 11, 13])
    axis.set_xlim([3, 13])

plt.tight_layout()

plt.savefig("figures/res-models-charge-silica.pdf", dpi=240)
plt.show()
