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
import plotting

rcParams["font.size"] = 8
rcParams["lines.linewidth"] = 0.75
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

PHI0_V = -1
# DEFAULT_GAMMA = (5.2/3.1) ** 3
gamma_list = plotting.GAMMA_LIST
conc_list = plotting.CONC_LIST

# potentials_v_she = np.linspace(-1.5, 0.5, 100)
potentials_v_pzc = np.linspace(-2, 0, 100)

# figure setup
fig = plt.figure(figsize=(6.69423, 4))
gs = GridSpec(nrows=2, ncols=3)
ax_conc_profile = fig.add_subplot(gs[0, 0])
ax_gamm_profile = fig.add_subplot(gs[1, 0])
ax_conc = fig.add_subplot(gs[0, 1])
ax_gamm = fig.add_subplot(gs[1, 1])
ax_pt_conc = fig.add_subplot(gs[0, 2])
ax_pt_gamm = fig.add_subplot(gs[1, 2])

colors1 = plotting.get_color_gradient(len(conc_list))
colors2 = plotting.get_color_gradient(len(gamma_list), color="red")

# potential profiles
for i, conc in enumerate(conc_list):
    gon = models.DoubleLayerModel(conc, DEFAULT_GAMMA, 2, 0, 2)
    solution = gon.spatial_profiles(PHI0_V, p_h=7, tol=1e-4)
    ax_conc_profile.plot(
        solution["x"],
        solution["phi"],
        color=colors1[i],
        label=f"{conc*1e3:.0f}",
    )

for i, gamma in enumerate(gamma_list):
    gon = models.DoubleLayerModel(plotting.DEFAULT_CONC_M, gamma, 2, 0, 2)
    solution = gon.spatial_profiles(PHI0_V, p_h=7, tol=1e-4)
    ax_gamm_profile.plot(
        solution["x"], solution["phi"], color=colors2[i], label=f"{gamma:.0f}"
    )


ax_conc_profile.set_ylim([-1, 0.01])
ax_conc_profile.set_xlim([0, 1])
ax_conc_profile.set_ylabel(r"$\phi$ / V")
ax_conc_profile.set_xlabel(r"$x$ / nm")
ax_conc_profile.legend(frameon=False, title=r"$c^*$ / mM")

ax_gamm_profile.set_ylim([-1, 0.01])
ax_gamm_profile.set_xlim([0, 1])
ax_gamm_profile.set_ylabel(r"$\phi$ / V")
ax_gamm_profile.set_xlabel(r"$x$ / nm")
ax_gamm_profile.legend(frameon=False, title=r"$\gamma_+$")

# driving forces for gamma
gamma_sol_list = []

for gamma in gamma_list:
    model = models.DoubleLayerModel(plotting.DEFAULT_CONC_M, gamma, 2, 0, 2)
    sol = model.potential_sweep(potentials_v_pzc, tol=1e-4, p_h=11)
    gamma_sol_list.append(sol)

for i, gamma in enumerate(gamma_list):
    ax_gamm.plot(
        potentials_v_pzc + C.AU_PZC_SHE_V,
        gamma_sol_list[i]["phi0"] - gamma_sol_list[i]["phi_rp"],
        color=colors2[i],
        label=f"{gamma:.0f}",
    )
    ax_pt_gamm.plot(
        potentials_v_pzc + C.PT_PZC_SHE_V,
        gamma_sol_list[i]["phi_rp"],
        color=colors2[i],
        label=f"{gamma:.0f}",
    )

ax_gamm.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
ax_gamm.set_xlim([-1.5, 0.5])
ax_gamm.set_ylabel(r"$\phi_0 - \phi'$ / V")
ax_gamm.set_ylim([-1.6, 0])
# ax_gamm.legend(frameon=False, title=r"$\gamma_+$")

ax_pt_gamm.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
ax_pt_gamm.set_xlim([-1.5, 0.5])
ax_pt_gamm.set_ylabel(r"$\phi'$ / V")
ax_pt_gamm.set_ylim([-0.6, 0])
# ax_pt_gamm.legend(frameon=False, title=r"$\gamma_+$")

# driving forces for conc
conc_sol_list = []

for conc in conc_list:
    model = models.DoubleLayerModel(conc, DEFAULT_GAMMA, 2, 0, 2)
    sol = model.potential_sweep(potentials_v_pzc, tol=1e-4, p_h=11)
    conc_sol_list.append(sol)

for i, conc in enumerate(conc_list):
    ax_conc.plot(
        potentials_v_pzc + C.AU_PZC_SHE_V,
        conc_sol_list[i]["phi0"] - conc_sol_list[i]["phi_rp"],
        color=colors1[i],
        label=f"{conc*1e3:.0f}",
    )
    ax_pt_conc.plot(
        potentials_v_pzc + C.PT_PZC_SHE_V,
        conc_sol_list[i]["phi_rp"],
        color=colors1[i],
        label=f"{conc*1e3:.0f}",
    )


ax_conc.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
ax_conc.set_xlim([-1.5, 0.5])
ax_conc.set_ylabel(r"$\phi_0 - \phi'$ / V")
ax_conc.set_ylim([-1.6, 0])
# ax_conc.legend(frameon=False, title=r"$c^*$ / mM")

ax_pt_conc.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
ax_pt_conc.set_xlim([-1.5, 0.5])
ax_pt_conc.set_ylabel(r"$\phi'$ / V")
ax_pt_conc.set_ylim([-0.6, 0])
# ax_pt_conc.legend(frameon=False, title=r"$c^*$ / mM")


# subfigure labelling
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
    # axis.set_xlabel(r"$\mathsf{E} - \mathsf{E}_\mathrm{pzc}$ / V")

plt.tight_layout()

plt.savefig("figures/res-driving-force.pdf")
# import tikzplotlib
# tikzplotlib.clean_figure()
# tikzplotlib.save("figures/res-driving-forces.tex", axis_width = '\\linewidth')
plt.show()
