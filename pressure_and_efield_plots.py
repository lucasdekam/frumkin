"""
Making figures for pressure and efield plots
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.legend_handler as lh
import matplotlib.transforms as mtransforms
from matplotlib import rcParams

import complete_edl as edl
import constants as C
import plotting


# plt.style.use("classic")
# rcParams["font.family"] = ["serif"]
# rcParams["font.weight"] = 300
# rcParams["mathtext.fontset"] = "dejavuserif"
rcParams["lines.linewidth"] = 1

PH_WERT = 13
POTENTIALS_V_RHE = np.linspace(-1, 0, 100)
POTENTIALS = POTENTIALS_V_RHE - C.AU_PZC_SHE_V - 59e-3 * PH_WERT
DEFAULT_GAMMA = 4
DEFAULT_CONC = 1e-2  # 10 mM
GAMMA_RANGE = [3, 4, 5, 6]
CONC_RANGE = [10e-3, 100e-3, 1000e-3]

p_au_gamma = np.zeros((len(GAMMA_RANGE), len(POTENTIALS)))
e_au_gamma = np.zeros((len(GAMMA_RANGE), len(POTENTIALS)))
p_si_gamma = np.zeros((len(GAMMA_RANGE), len(POTENTIALS)))
e_si_gamma = np.zeros((len(GAMMA_RANGE), len(POTENTIALS)))

# Gamma sweep
for i, gamma in enumerate(GAMMA_RANGE):
    gold = edl.Aqueous(DEFAULT_CONC, gamma, DEFAULT_GAMMA, DEFAULT_GAMMA, 3)
    gold_sol = gold.potential_sweep(POTENTIALS, p_h=PH_WERT)
    p_au_gamma[i, :] = gold_sol["pressure"]
    e_au_gamma[i, :] = gold_sol["efield"]

    ins = edl.Aqueous(DEFAULT_CONC, gamma, DEFAULT_GAMMA, DEFAULT_GAMMA, 3)
    ins_sol = ins.insulator_spatial_profiles(p_h=PH_WERT, tol=1e-2)
    p_si_gamma[i, :] = np.ones(POTENTIALS.shape) * ins_sol["pressure"][0]
    e_si_gamma[i, :] = np.ones(POTENTIALS.shape) * ins_sol["efield"][0]

color_green = plotting.get_color_gradient(len(GAMMA_RANGE), color="green")
color_blue = plotting.get_color_gradient(len(GAMMA_RANGE), color="blue")

p_au_conc = np.zeros((len(CONC_RANGE), len(POTENTIALS)))
p_si_conc = np.zeros((len(CONC_RANGE), len(POTENTIALS)))
e_au_conc = np.zeros((len(CONC_RANGE), len(POTENTIALS)))
e_si_conc = np.zeros((len(CONC_RANGE), len(POTENTIALS)))

# Conc sweep
for i, conc in enumerate(CONC_RANGE):
    gold = edl.Aqueous(conc, DEFAULT_GAMMA, DEFAULT_GAMMA, DEFAULT_GAMMA, 3)
    gold_sol = gold.potential_sweep(POTENTIALS, p_h=PH_WERT)
    p_au_conc[i, :] = gold_sol["pressure"]
    e_au_conc[i, :] = gold_sol["efield"]

    ins = edl.Aqueous(conc, DEFAULT_GAMMA, DEFAULT_GAMMA, DEFAULT_GAMMA, 3)
    ins_sol = ins.insulator_spatial_profiles(p_h=PH_WERT, tol=1e-2)
    p_si_conc[i, :] = np.ones(POTENTIALS.shape) * ins_sol["pressure"][0]
    e_si_conc[i, :] = np.ones(POTENTIALS.shape) * ins_sol["efield"][0]

# Figure creation
fig, ax = plt.subplots(
    figsize=(9, 6),
    nrows=2,
    ncols=2,
    # sharex=True,
    sharey="col",
    facecolor="1",
)

# Gamma sweep: Pressure and efield plots
greens = plotting.get_color_gradient(len(GAMMA_RANGE), color="green")
blues = plotting.get_color_gradient(len(GAMMA_RANGE), color="blue")

p_gold_entries = []
p_si_entries = []
e_gold_entries = []
e_si_entries = []

for i, gamma in enumerate(GAMMA_RANGE):
    (gp,) = ax[0, 1].plot(
        POTENTIALS_V_RHE,
        p_au_gamma[i, :] / 1e5,
        color=blues[i],
    )
    (sp,) = ax[0, 1].plot(
        POTENTIALS_V_RHE,
        p_si_gamma[i, :] / 1e5,
        color=greens[i],
        # label=r"$\gamma_{AM}=$" + f"{gamma}",
    )
    p_gold_entries.append(gp)
    p_si_entries.append(sp)

    (ge,) = ax[0, 0].plot(
        POTENTIALS_V_RHE,
        e_au_gamma[i, :] * 1e-9,
        color=blues[i],
    )
    (se,) = ax[0, 0].plot(
        POTENTIALS_V_RHE,
        e_si_gamma[i, :] * 1e-9,
        color=greens[i],
        # label=r"$\gamma_{AM}=$" + f"{gamma}",
    )
    e_gold_entries.append(ge)
    e_si_entries.append(se)

    ax[0, 1].set_ylabel(r"$P$ [bar]")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylim([1e3, 1e5])
    ax[0, 0].set_xlim([-1, 0])
    ax[0, 1].set_xlim([-1, 0])
    ax[0, 0].set_xlabel(r"$\phi_0$ [V vs. RHE]")
    ax[0, 1].set_xlabel(r"$\phi_0$ [V vs. RHE]")
    ax[0, 0].set_ylabel(r"$\mathcal{E}$ [V/nm]")
    ax[0, 0].set_ylim([-7, 0])

box = ax[0, 1].get_position()
ax[0, 1].set_position([box.x0, box.y0, box.width * 0.6, box.height])
leg2 = ax[0, 1].legend(
    [(g, s) for (g, s) in zip(e_gold_entries, e_si_entries)],
    [r"$\gamma_{AM}=$" + f"{gamma}" for gamma in GAMMA_RANGE],
    handler_map={tuple: lh.HandlerTuple(ndivide=None)},
    fontsize="medium",
    handlelength=5,
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    title="  Au | SiN",
    edgecolor="white",
    fancybox=False,
    alignment="left",
)
ax[0, 1].add_artist(leg2)

# Conc sweep: pressure and efield plots
greens = plotting.get_color_gradient(len(CONC_RANGE), color="green")
blues = plotting.get_color_gradient(len(CONC_RANGE), color="blue")

p_gold_entries = []
p_si_entries = []
e_gold_entries = []
e_si_entries = []

for i, conc in enumerate(CONC_RANGE):
    (gp,) = ax[1, 1].plot(
        POTENTIALS_V_RHE,
        p_au_conc[i, :] / 1e5,
        color=blues[i],
    )
    (sp,) = ax[1, 1].plot(
        POTENTIALS_V_RHE,
        p_si_conc[i, :] / 1e5,
        color=greens[i],
    )
    p_gold_entries.append(gp)
    p_si_entries.append(sp)

    (ge,) = ax[1, 0].plot(
        POTENTIALS_V_RHE,
        e_au_conc[i, :] * 1e-9,
        color=blues[i],
    )
    (se,) = ax[1, 0].plot(
        POTENTIALS_V_RHE,
        e_si_conc[i, :] * 1e-9,
        color=greens[i],
    )
    e_gold_entries.append(ge)
    e_si_entries.append(se)

    ax[1, 1].set_ylabel(r"$P$ [bar]")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylim([1e3, 1e5])
    ax[1, 0].set_xlim([-1, 0])
    ax[1, 1].set_xlim([-1, 0])
    ax[1, 0].set_xlabel(r"$\phi_0$ [V vs. RHE]")
    ax[1, 1].set_xlabel(r"$\phi_0$ [V vs. RHE]")
    ax[1, 0].set_ylabel(r"$\mathcal{E}$ [V/nm]")

box = ax[1, 1].get_position()
ax[1, 1].set_position([box.x0, box.y0, box.width * 0.6, box.height])
leg2 = ax[1, 1].legend(
    [(g, s) for (g, s) in zip(e_gold_entries, e_si_entries)],
    [r"$c_{AM}^b=$" + f"{conc*1000:.0f}mM" for conc in CONC_RANGE],
    handler_map={tuple: lh.HandlerTuple(ndivide=None)},
    fontsize="medium",
    handlelength=5,
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    title="  Au | SiN",
    edgecolor="white",
    fancybox=False,
    alignment="left",
)
ax[1, 1].add_artist(leg2)

labels = ["(a)", "(b)", "(c)", "(d)"]
for label, axis in zip(labels, ax.reshape(-1)):
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

plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig("figures/nano-pressures-efields.pdf")
plt.show()
