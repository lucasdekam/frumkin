"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
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

PHI0_V = -1
GAMMA = 6
N_W = C.C_WATER_BULK * 1e3 * C.N_A
# A_M = 10e-10
A_M = 6e-10  # (GAMMA / N_W) ** (1/3)
print(A_M)


def stern(x, slope):  # pylint: disable=invalid-name
    """
    Potential profile in Stern layer
    """
    return PHI0_V - slope * x


potentials = np.linspace(-1, 1, 100)
concentration_range = CONC_LIST = [1e-3, 1e-2, 1e-1, 2]

sols = []
sweeps = []

for i, conc in enumerate(concentration_range):
    model = models.Borukhov(conc, A_M)
    sweep = model.potential_sweep(potentials, tol=1e-3)
    sweeps.append(sweep)
    # capa_phi_sweep[i, :] = sweep["capacity"]
    # sigma_phi_sweep[i, :] = sweep["charge"]
    # eps_phi_sweep[i, :] = sweep["eps"]

    spatial = model.spatial_profiles(phi0=PHI0_V, tol=1e-3)
    spatial["x_shifted"] = spatial["x"] + A_M / 2 * 1e9
    sols.append(spatial)

# fig, ax = plt.subplots(figsize=(5, 4), nrows=2, ncols=2)
fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(223)

colors1 = plotting.get_color_gradient(len(concentration_range))
colors2 = plotting.get_color_gradient(len(concentration_range), color="red")

# c_entries = []
# a_entries = []

for i, conc in enumerate(concentration_range):
    ax1.plot(
        sols[i]["x_shifted"],
        sols[i]["phi"],
        label=f"{conc*1e3:.0f}",
        color=colors1[i],
    )
    x_m = np.linspace(0, A_M / 2, 50)

    ax1.plot(
        x_m * 1e9,
        stern(x_m, sols[i]["efield"][0]),
        color=colors1[i],
    )

    ax2.plot(
        sols[i]["x_shifted"],
        sols[i]["cations"],
        label=f"{conc*1e3:.0f} mM",
        color=colors1[i],
    )

    # ax3.plot(
    #     sols[i]['x_shifted'],
    #     sols[i]['eps'],
    #     label=f"{conc*1e3:.0f} mM",
    #     color=colors1[i],
    # )

    # ax3.plot(
    #     x_m * 1e9,
    #     np.ones(x_m.shape) * sols[i]['eps'][0],
    #     color=colors1[i]
    # )

    ax4.plot(
        potentials,
        sweeps[i]["capacity"] * 100,
        label=f"{conc*1e3:.0f} mM",
        color=colors1[i],
    )

ax1.set_ylabel(r"$\phi$ / V")
ax2.set_ylabel(r"$c_+$ / M")
# ax3.set_ylabel(r"$\varepsilon/\varepsilon_0$")
ax4.set_ylabel(r"$C$ / $\mu$F cm$^{-2}$")

ax1.set_ylim([PHI0_V, 0.05])
ax2.set_ylim([0, 10])
# ax3.set_ylim([0, 80])
ax4.set_ylim([0, 150])

ax1.set_xlabel(r"$x$ / nm")
ax2.set_xlabel(r"$x$ / nm")
# ax3.set_xlabel(r"$x$ / nm")
ax4.set_xlabel(r"$\phi_0$ / V")

ax1.set_xlim([0, 2])
ax2.set_xlim([0, 5])
# ax3.set_xlim([0, 5])
ax4.set_xlim([potentials[0], potentials[-1]])

# ax[0, 0].set_xticks([0, 1, 2, 3, 4, 5])

ax1.legend(frameon=False, title=r"$c_0$ / mM")

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

plt.tight_layout()
plt.savefig("figures/intro-bikerman.pdf")

plt.show()
