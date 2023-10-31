"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

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

PHI0_V = -1
concentration_range = plotting.CONC_LIST
gamma_range = plotting.GAMMA_LIST
phi0_v_range = [-0.4, -0.8, -1.2, -1.6, -2.0]

gcs = models.GouyChapmanStern(plotting.DEFAULT_CONC_M, A_M / 2)
lpb = models.LangevinPoissonBoltzmann(plotting.DEFAULT_CONC_M, A_M / 2)
bik = models.Borukhov(plotting.DEFAULT_CONC_M, A_M)
gon = models.AqueousVariableStern(plotting.DEFAULT_CONC_M, GAMMA, 2, 4, 4)
sol_gcs = gcs.spatial_profiles(PHI0_V, tol=1e-4)
sol_lpb = lpb.spatial_profiles(PHI0_V, tol=1e-4)
sol_bik = bik.spatial_profiles(PHI0_V, tol=1e-4)
sol_gon = gon.spatial_profiles(PHI0_V, tol=1e-4)

gi_conc = []
gi_gamm = []
gi_phi0 = []

for i, conc in enumerate(concentration_range):
    gon = models.AqueousVariableStern(conc, GAMMA, 2, 4, 4)
    solution = gon.spatial_profiles(PHI0_V, p_h=7, tol=1e-4)
    gi_conc.append(solution)

for i, gamma in enumerate(gamma_range):
    gon = models.AqueousVariableStern(plotting.DEFAULT_CONC_M, gamma, 2, 4, 4)
    solution = gon.spatial_profiles(PHI0_V, p_h=7, tol=1e-4)
    gi_gamm.append(solution)

for i, phi0_v in enumerate(phi0_v_range):
    gon = models.AqueousVariableStern(plotting.DEFAULT_CONC_M, GAMMA, 2, 4, 4)
    solution = gon.spatial_profiles(phi0_v, p_h=7, tol=1e-4)
    gi_phi0.append(solution)

# fig, ax = plt.subplots(figsize=(5, 4), nrows=2, ncols=2)
fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
axes = [ax1, ax2, ax3, ax4]


ax1.plot(sol_gcs["x"], sol_gcs["cations"], "k--", label="GCS")
ax1.plot(sol_lpb["x"], sol_lpb["cations"], "k:", label="LPB")
ax1.plot(sol_bik["x"], sol_bik["cations"], "k-.", label="Bikerman")
ax1.plot(sol_gon["x"], sol_gon["cations"], "k", label="GI")
ax1.set_xlim([0, 5])
ax1.legend(frameon=False)

colors1 = plotting.get_color_gradient(len(concentration_range))
colors2 = plotting.get_color_gradient(len(gamma_range), color="red")
colors3 = plotting.get_color_gradient(len(phi0_v_range), color="purple")

for i, conc in enumerate(concentration_range):
    ax2.plot(
        gi_conc[i]["x"],
        gi_conc[i]["cations"],
        color=colors1[i],
        label=f"{conc*1e3:.0f}",
    )
ax2.legend(frameon=False, title=r"$c_+^*$ / mM")

for i, gamma in enumerate(gamma_range):
    ax3.plot(
        gi_gamm[i]["x"], gi_gamm[i]["cations"], color=colors2[i], label=f"{gamma:.0f}"
    )
ax3.legend(frameon=False, title=r"$\gamma_+$")

for i, phi0_v in enumerate(phi0_v_range):
    ax4.plot(
        gi_phi0[i]["x"], gi_phi0[i]["cations"], color=colors3[i], label=f"{phi0_v:.1f}"
    )
ax4.legend(frameon=False, title=r"$\phi_0$ / V")


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
    axis.set_ylim([0, 8])
    axis.set_xlim([0, 2])
    axis.set_ylabel(r"$c_+$ / M")
    axis.set_xlabel(r"$x$ / nm")
ax1.set_ylim([0, 10])
ax1.set_xlim([0, 5])

plt.tight_layout()
plt.savefig("figures/res-cation-comparison.pdf")
plt.show()
