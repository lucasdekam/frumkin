"""
Making figures for current simulation plots
"""
import numpy as np

import matplotlib.pyplot as plt
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

DEFAULT_GAMMA = 4
DEFAULT_CONC = 1e-1  # 100 mM
GAMMA_RANGE = [3, 4, 5, 6]
CONC_RANGE = [10e-3, 100e-3, 1000e-3]

C_1 = 6
C_2 = 3


def calculate_current(gold_sol, silanol_sol):
    """
    Calculate the current from a Aqueous models solved for gold and insulator
    """
    frac_sioh = (
        C.N_SITES_SILICA
        + silanol_sol["efield"][0] * silanol_sol["eps"][0] * C.EPS_0 / C.E_0
    ) / C.N_SITES_SILICA
    reorganization = C_1 * C.E_0 + np.abs(silanol_sol["pressure"][0]) * C_2 / au.n_max
    reaction_energy = C.E_0 * gold_sol["phi0"]
    e_act = (reorganization + reaction_energy) ** 2 / (4 * reorganization)
    return -frac_sioh * np.exp(-C.BETA * e_act)


PH_WERT_GAMM = 13
potentials_v_rhe_gamm = np.linspace(-0.5, 0, 100)
potentials_gamm = potentials_v_rhe_gamm - C.AU_PZC_SHE_V - 59e-3 * PH_WERT_GAMM
current_gamm = np.zeros((len(GAMMA_RANGE), len(potentials_gamm)))
colors_gamm = plotting.get_color_gradient(len(GAMMA_RANGE))

for i, gamma in enumerate(GAMMA_RANGE):
    au = edl.Aqueous(DEFAULT_CONC, gamma, DEFAULT_GAMMA, DEFAULT_GAMMA, C_2)
    au_sol = au.potential_sweep(potentials_gamm, p_h=PH_WERT_GAMM)
    silica = edl.Aqueous(DEFAULT_CONC, gamma, DEFAULT_GAMMA, DEFAULT_GAMMA, C_2)
    silica_sol = silica.insulator_spatial_profiles(p_h=PH_WERT_GAMM, tol=1e-2)

    current_gamm[i, :] = calculate_current(au_sol, silica_sol)
current_gamm = current_gamm / np.max(np.abs(current_gamm))

PH_WERT_CONC = 7
potentials_v_rhe_conc = np.linspace(-0.25, 0, 100)
potentials_conc = potentials_v_rhe_conc - C.AU_PZC_SHE_V - 59e-3 * PH_WERT_CONC
current_conc = np.zeros((len(CONC_RANGE), len(potentials_conc)))
colors_conc = plotting.get_color_gradient(len(CONC_RANGE))

for i, conc in enumerate(CONC_RANGE):
    au = edl.Aqueous(conc, DEFAULT_GAMMA, DEFAULT_GAMMA, DEFAULT_GAMMA, C_2)
    au_sol = au.potential_sweep(potentials_conc, p_h=PH_WERT_CONC)
    silica = edl.Aqueous(conc, DEFAULT_GAMMA, DEFAULT_GAMMA, DEFAULT_GAMMA, C_2)
    silica_sol = silica.insulator_spatial_profiles(p_h=PH_WERT_CONC, tol=1e-2)
    current_conc[i, :] = calculate_current(au_sol, silica_sol)
current_conc = current_conc / np.max(np.abs(current_conc))

fig, ax = plt.subplots(figsize=(9, 3), nrows=1, ncols=2)
for i, gamma in enumerate(GAMMA_RANGE):
    ax[0].plot(
        potentials_v_rhe_gamm,
        current_gamm[i, :],
        color=colors_gamm[i],
        label=r"$\gamma_{AM}=$" + f"{gamma}",
    )

for i, conc in enumerate(CONC_RANGE):
    ax[1].plot(
        potentials_v_rhe_conc,
        np.log10(-current_conc[i, :]),
        color=colors_conc[i],
        label=r"$c_{AM}^b=$" + f"{conc*1000:.0f}mM",
    )

labels = ["a)", "b)"]
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


ax[0].set_ylabel(r"$j/|j_\mathrm{max}|$")
ax[0].set_xlabel(r"$\phi_0$ [V vs. RHE]")
ax[0].legend(frameon=False, title=f"pH {PH_WERT_GAMM}")
ax[1].set_ylabel(r"$\log_{10} (|j|/|j_\mathrm{max}|)$")
ax[1].set_xlabel(r"$\phi_0$ [V vs. RHE]")
ax[1].legend(frameon=False, title=f"pH {PH_WERT_CONC}")
plt.tight_layout()
plt.savefig("figures/nano_currents.pdf")
plt.show()
