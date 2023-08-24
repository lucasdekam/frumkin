"""
Making figures for fraction SiOH plots
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
PH_RANGE = np.linspace(2, 13, 200)


def calculate_f_sioh(ph_sweep_sol):
    """
    Calculate f_SiOH
    """
    frac_sioh = (C.N_SITES_SILICA + ph_sweep_sol["charge"] / C.E_0) / C.N_SITES_SILICA
    return frac_sioh


ph_range = np.linspace(2, 13, 200)
frac_sioh_gamm = np.zeros((len(GAMMA_RANGE), len(PH_RANGE)))

for i, gamma in enumerate(GAMMA_RANGE):
    silica = edl.Aqueous(DEFAULT_CONC, gamma, DEFAULT_GAMMA, DEFAULT_GAMMA, 3)
    silica_sol = silica.ph_sweep(ph_range, tol=1e-2)
    frac_sioh_gamm[i, :] = calculate_f_sioh(silica_sol)

frac_sioh_conc = np.zeros((len(CONC_RANGE), len(ph_range)))
for i, conc in enumerate(CONC_RANGE):
    silica = edl.Aqueous(conc, DEFAULT_GAMMA, DEFAULT_GAMMA, DEFAULT_GAMMA, 3)
    silica_sol = silica.ph_sweep(ph_range, tol=1e-2)
    frac_sioh_conc[i, :] = (
        C.N_SITES_SILICA + silica_sol["charge"] / C.E_0
    ) / C.N_SITES_SILICA

fig, ax = plt.subplots(figsize=(9, 3), nrows=1, ncols=2, sharex=True)
colors_gamm = plotting.get_color_gradient(len(GAMMA_RANGE), color="green")
colors_conc = plotting.get_color_gradient(len(CONC_RANGE), color="green")
for i, gamma in enumerate(GAMMA_RANGE):
    ax[0].plot(
        ph_range,
        frac_sioh_gamm[i, :],
        color=colors_gamm[i],
        label=r"$\gamma_{AM}=$" + f"{gamma}",
    )
for i, conc in enumerate(CONC_RANGE):
    ax[1].plot(
        ph_range,
        frac_sioh_conc[i, :],
        color=colors_conc[i],
        label=r"$c_{AM}^b=$" + f"{conc*1000:.0f}mM",
    )

ax[0].set_ylabel(r"$f_\mathrm{SiOH}$")
ax[0].set_xlabel(r"pH")
ax[0].legend(frameon=False)
ax[0].set_ylim([0.5, 1])
ax[0].set_xlim([2, 13])
ax[0].set_xticks(np.arange(3, 13 + 2, 2))

ax[1].set_xticks(np.arange(3, 13 + 2, 2))
ax[1].set_ylabel(r"$f_\mathrm{SiOH}$")
ax[1].set_xlabel(r"pH")
ax[1].legend(frameon=False)
ax[1].set_ylim([0.5, 1])

labels = ["(a)", "(b)"]
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

plt.tight_layout()
plt.savefig("figures/nano_sioh.pdf")
plt.show()
