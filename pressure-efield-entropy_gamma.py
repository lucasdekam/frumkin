"""
Making figures for pressure and efield plots
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.legend_handler as lh
import matplotlib.transforms as mtransforms
from matplotlib import rcParams

from edl import models
from edl import constants as C
import plotting


rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


PH_WERT = 13
POTENTIALS_V_RHE = np.linspace(-1, C.AU_MAIER_PZC_SHE_V + 59e-3 * PH_WERT, 100)
POTENTIALS = POTENTIALS_V_RHE - C.AU_MAIER_PZC_SHE_V - 59e-3 * PH_WERT
DEFAULT_CONC = 1e-2  # 10 mM
GAMMA_RANGE = plotting.GAMMA_LIST

# Conc sweep
p_au = np.zeros((len(GAMMA_RANGE), len(POTENTIALS)))
p_si = np.zeros((len(GAMMA_RANGE), len(POTENTIALS)))
e_au = np.zeros((len(GAMMA_RANGE), len(POTENTIALS)))
e_si = np.zeros((len(GAMMA_RANGE), len(POTENTIALS)))
s_au = np.zeros((len(GAMMA_RANGE), len(POTENTIALS)))
s_si = np.zeros((len(GAMMA_RANGE), len(POTENTIALS)))

for i, gamma in enumerate(GAMMA_RANGE):
    model = models.AqueousVariableStern(DEFAULT_CONC, gamma, 2, 4, 4)
    gold_sol = model.potential_sweep(POTENTIALS, p_h=PH_WERT)
    p_au[i, :] = gold_sol["pressure"]
    e_au[i, :] = gold_sol["efield"]
    s_au[i, :] = gold_sol["entropy"]

    ins_sol = model.insulator_spatial_profiles(p_h=PH_WERT, tol=1e-2)
    p_si[i, :] = np.ones(POTENTIALS.shape) * ins_sol["pressure"][0]
    e_si[i, :] = np.ones(POTENTIALS.shape) * ins_sol["efield"][0]
    s_si[i, :] = np.ones(POTENTIALS.shape) * ins_sol["entropy"][0]

# Figure creation
fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(221)
ax3 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)

# Conc sweep: pressure and efield plots
greens = plotting.get_color_gradient(len(GAMMA_RANGE), color="green")
reds = plotting.get_color_gradient(len(GAMMA_RANGE), color="red")

p_gold_entries = []
p_si_entries = []
e_gold_entries = []
e_si_entries = []
s_gold_entries = []
s_si_entries = []

for i, gamma in enumerate(GAMMA_RANGE):
    (gp,) = ax2.plot(
        POTENTIALS_V_RHE,
        p_au[i, :] / 1e5,
        color=reds[i],
    )
    (sp,) = ax2.plot(
        POTENTIALS_V_RHE,
        p_si[i, :] / 1e5,
        color=greens[i],
    )
    p_gold_entries.append(gp)
    p_si_entries.append(sp)

    (ge,) = ax1.plot(
        POTENTIALS_V_RHE,
        e_au[i, :] * 1e-9,
        color=reds[i],
    )
    (se,) = ax1.plot(
        POTENTIALS_V_RHE,
        e_si[i, :] * 1e-9,
        color=greens[i],
    )
    e_gold_entries.append(ge)
    e_si_entries.append(se)

    (ge,) = ax3.plot(
        POTENTIALS_V_RHE,
        s_au[i, :] / C.C_WATER_BULK / C.N_A / 1e3,
        color=reds[i],
    )
    (se,) = ax3.plot(
        POTENTIALS_V_RHE,
        s_si[i, :] / C.C_WATER_BULK / C.N_A / 1e3,
        color=greens[i],
    )

ax1.set_ylabel(r"$E$ / V nm$^{-1}$")
ax2.set_ylabel(r"$P(0)$ / bar")
ax1.set_ylim([-5, -0.5])
ax2.set_ylim([0, 15000])
ax3.set_ylim([-1.3, -0.2])

ax3.set_ylabel(r"$s(0)a^3/k_\mathrm{B}$")
# box = ax2.get_position()
# ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])
leg2 = ax1.legend(
    [(g, s) for (g, s) in zip(e_gold_entries, e_si_entries)],
    [f"{gamma:.0f}" for gamma in GAMMA_RANGE],
    handler_map={tuple: lh.HandlerTuple(ndivide=None)},
    # fontsize="medium",
    handlelength=2.5,
    # loc="center left",
    # bbox_to_anchor=(1.05, 0.5),
    # edgecolor="white",
    # fancybox=False,
    frameon=False,
    alignment="left",
    # title=r"$c_0$",
)
ax1.add_artist(leg2)

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
    axis.set_xlabel(r"$\mathsf{E}$ / V vs. RHE")
    axis.set_xlim([-1, 0])

# plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.tight_layout()
plt.savefig("figures/nano-gamma.pdf")
plt.show()
