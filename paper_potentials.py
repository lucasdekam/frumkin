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

rcParams["font.size"] = 8
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
fig_phi = plt.figure(figsize=(6.69423, 4))
gs = GridSpec(nrows=2, ncols=2)
ax_phi_cbulk = fig_phi.add_subplot(gs[0, 0])
ax_phi_gamma = fig_phi.add_subplot(gs[1, 0])
ax_phirp_cbulk = fig_phi.add_subplot(gs[0, 1])
ax_phirp_gamma = fig_phi.add_subplot(gs[1, 1])

fig_conc = plt.figure(figsize=(3.248, 4))
ax_cat_cbulk = fig_conc.add_subplot(211)
ax_oh_phbulk = fig_conc.add_subplot(212)

colors1 = plotting.get_color_gradient(len(cbulk_list))
colors2 = plotting.get_color_gradient(len(gamma_list), color="red")
colors3 = plotting.get_color_gradient(len(ph_list), color="purple")

# potential profiles
for i, conc in enumerate(cbulk_list):
    gon = models.DoubleLayerModel(conc, DEFAULT_GAMMA, 2)
    solution = gon.spatial_profiles(PHI0_V, tol=1e-4)
    ax_phi_cbulk.plot(
        solution["x"],
        solution["phi"],
        color=colors1[i],
        label=f"{conc*1e3:.0f}",
    )

for i, gamma in enumerate(gamma_list):
    gon = models.DoubleLayerModel(DEFAULT_CONC, gamma, 2)
    solution = gon.spatial_profiles(PHI0_V, tol=1e-4)
    ax_phi_gamma.plot(
        solution["x"], solution["phi"], color=colors2[i], label=f"{gamma:.0f}"
    )


ax_phi_cbulk.set_ylim([-1, 0.01])
ax_phi_cbulk.set_xlim([0, 1])
ax_phi_cbulk.set_ylabel(r"$\phi$ / V")
ax_phi_cbulk.set_xlabel(r"$x$ / nm")
ax_phi_cbulk.legend(frameon=False, title=r"$c_+^*$ / mM")

ax_phi_gamma.set_ylim([-1, 0.01])
ax_phi_gamma.set_xlim([0, 1])
ax_phi_gamma.set_ylabel(r"$\phi$ / V")
ax_phi_gamma.set_xlabel(r"$x$ / nm")
ax_phi_gamma.legend(frameon=False, title=r"$\gamma_+$")

# driving forces for gamma
gamma_sol_list = []

for gamma in gamma_list:
    model = models.DoubleLayerModel(DEFAULT_CONC, gamma, 2)
    sol = model.potential_sweep(potentials_v_pzc, tol=1e-4)
    gamma_sol_list.append(sol)

phirp_entries = []
for i, gamma in enumerate(gamma_list):
    (p,) = ax_phirp_gamma.plot(
        potentials_v_pzc + C.AU_PZC_SHE_V,
        gamma_sol_list[i]["phi_rp"],
        color=colors2[i],
    )
    phirp_entries.append(p)

(phi0p,) = ax_phirp_gamma.plot(
    potentials_v_pzc + C.AU_PZC_SHE_V,
    potentials_v_pzc,
    "k",
    label=r"$\phi_0$",
)

ax_phirp_gamma.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
ax_phirp_gamma.set_xlim([-1.5, 0.5])
ax_phirp_gamma.set_ylabel(r"$\phi$ / V")
ax_phirp_gamma.set_ylim([-2, 0])
ax_phirp_gamma.legend(
    [tuple(phirp_entries), (phi0p,)],
    [r"$\phi'$", r"$\phi_0$"],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
    frameon=False,
)

# driving forces for conc
conc_sol_list = []

for conc in cbulk_list:
    model = models.DoubleLayerModel(conc, DEFAULT_GAMMA, 2)
    sol = model.potential_sweep(potentials_v_pzc, tol=1e-4)
    conc_sol_list.append(sol)

phirp_entries = []
for i, conc in enumerate(cbulk_list):
    (p,) = ax_phirp_cbulk.plot(
        potentials_v_pzc + C.AU_PZC_SHE_V,
        conc_sol_list[i]["phi_rp"],
        color=colors1[i],
    )
    phirp_entries.append(p)

(phi0p,) = ax_phirp_cbulk.plot(
    potentials_v_pzc + C.AU_PZC_SHE_V,
    potentials_v_pzc,
    "k",
    label=r"$\phi_0$",
)

ax_phirp_cbulk.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
ax_phirp_cbulk.set_xlim([-1.5, 0.5])
ax_phirp_cbulk.set_ylabel(r"$\phi$ / V")
ax_phirp_cbulk.set_ylim([-2, 0])
ax_phirp_cbulk.legend(
    [tuple(phirp_entries), (phi0p,)],
    [r"$\phi'$", r"$\phi_0$"],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
    frameon=False,
)

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
ax_cat_cbulk.set_xlabel(r"E / V vs. SHE")
ax_cat_cbulk.legend(frameon=False, title=r"$c_+^*$ / mM")

# oh- concentration profiles
md = models.DoubleLayerModel(DEFAULT_CONC, DEFAULT_GAMMA, 2)
solution = md.potential_sweep(potentials_v_pzc, tol=1e-4)
for i, ph in enumerate(ph_list):
    oh_bulk = 10 ** (-14 + ph)
    oh_conc = solution["ani_2"].values / DEFAULT_CONC * oh_bulk
    ax_oh_phbulk.plot(
        potentials_v_pzc + C.AU_PZC_SHE_V,
        oh_conc,
        color=colors3[i],
        label=f"{ph:.0f}",
    )
    print(
        np.gradient(
            np.log(oh_conc) / C.BETA / C.E_0,
            potentials_v_pzc,
        )[0],
    )

ax_oh_phbulk.set_yscale("log")
ax_oh_phbulk.set_xlim([-1.5, 0.5])
ax_oh_phbulk.set_ylim([1e-25, 1])
ax_oh_phbulk.set_ylabel(r"$c_\mathrm{OH^-}(x_2)$ / M")
ax_oh_phbulk.set_xlabel(r"E / V vs. SHE")
ax_oh_phbulk.legend(frameon=False, title=r"pH", loc="lower right")

# subfigure labelling
labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
for label, axis in zip(labels, fig_phi.axes):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-25 / 72, 10 / 72, fig_phi.dpi_scale_trans)
    axis.text(
        0.0,
        1.0,
        label,
        transform=axis.transAxes + trans,
        fontsize="medium",
        va="bottom",
    )

for label, axis in zip(labels, fig_conc.axes):
    trans = mtransforms.ScaledTranslation(-25 / 72, 10 / 72, fig_phi.dpi_scale_trans)
    axis.text(
        0.0,
        1.0,
        label,
        transform=axis.transAxes + trans,
        fontsize="medium",
        va="bottom",
    )

fig_phi.tight_layout()
fig_phi.subplots_adjust(left=0.25, right=0.85, wspace=0.5)
fig_conc.tight_layout()
fig_conc.subplots_adjust(left=0.25, right=0.85)

fig_phi.savefig("figures/phi-profiles.pdf")
fig_conc.savefig("figures/concentration-epot.pdf")
plt.show()
