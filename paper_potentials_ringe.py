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
gamma_list = plotting.GAMMA_LIST
conc_list = plotting.CONC_LIST
DEFAULT_GAMMA = gamma_list[2]
X2_LIST = [3.5e-10, 4.1e-10, 5.2e-10, 5.8e-10]

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
    gon = models.ExplicitStern(conc, DEFAULT_GAMMA, 2, X2_LIST[2])
    solution = gon.spatial_profiles(PHI0_V, tol=1e-4)
    ax_conc_profile.plot(
        solution["x"],
        solution["phi"],
        color=colors1[i],
        label=f"{conc*1e3:.0f}",
    )

for i, gamma in enumerate(gamma_list):
    gon = models.ExplicitStern(plotting.DEFAULT_CONC_M, gamma, 2, X2_LIST[i])
    solution = gon.spatial_profiles(PHI0_V, tol=1e-4)
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

for i, gamma in enumerate(gamma_list):
    model = models.ExplicitStern(plotting.DEFAULT_CONC_M, gamma, 2, X2_LIST[i])
    sol = model.potential_sweep(potentials_v_pzc, tol=1e-4)
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
ax_gamm.set_ylim([-1.2, 0])
# ax_gamm.legend(frameon=False, title=r"$\gamma_+$")

ax_pt_gamm.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
ax_pt_gamm.set_xlim([-1.5, 0.5])
ax_pt_gamm.set_ylabel(r"$\phi'$ / V")
ax_pt_gamm.set_ylim([-1.2, 0])
# ax_pt_gamm.legend(frameon=False, title=r"$\gamma_+$")

# driving forces for conc
conc_sol_list = []

for conc in conc_list:
    model = models.ExplicitStern(conc, DEFAULT_GAMMA, 2, X2_LIST[2])
    sol = model.potential_sweep(potentials_v_pzc, tol=1e-4)
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
ax_conc.set_ylim([-1.2, 0])
# ax_conc.legend(frameon=False, title=r"$c^*$ / mM")

ax_pt_conc.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
ax_pt_conc.set_xlim([-1.5, 0.5])
ax_pt_conc.set_ylabel(r"$\phi'$ / V")
ax_pt_conc.set_ylim([-1.2, 0])
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

plt.tight_layout()

plt.savefig("figures/res-driving-force_ringe.pdf")
plt.show()
