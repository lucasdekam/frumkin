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

PHI0_V = -0.25
X_2 = 2.8e-10
CONC = 10e-3


def stern(x, slope):  # pylint: disable=invalid-name
    """
    Potential profile in Stern layer
    """
    return PHI0_V - slope * x


potentials = np.linspace(-1, 1, 100)
temp_range = np.array([5, 25, 50, 75]) + 273.15

solutions = []
sweeps = []

for i, temp in enumerate(temp_range):
    model = models.LangevinPoissonBoltzmann(CONC, x2=X_2, temp=temp, delta=-0.5)
    sweep_sol = model.potential_sweep(potentials, tol=1e-3)
    sweeps.append(sweep_sol)

    spatial_sol = model.spatial_profiles(phi0=PHI0_V, tol=1e-3)
    solutions.append(spatial_sol)


# fig, ax = plt.subplots(figsize=(5, 4), nrows=2, ncols=2)
fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
colors1 = plotting.get_color_gradient(len(temp_range))

for i, temp in enumerate(temp_range):
    ax1.plot(
        solutions[i]["x"],
        solutions[i]["phi"],
        label=f"{temp - 273.15:.0f}" + r"$^\circ$C",
        color=colors1[i],
    )

    ax2.plot(
        solutions[i]["x"],
        solutions[i]["cations"],
        color=colors1[i],
    )

    ax3.plot(
        potentials,
        sweeps[i]["eps"],
        color=colors1[i],
    )

    ax4.plot(
        potentials,
        sweeps[i]["capacity"] * 100,
        color=colors1[i],
    )

ax1.set_ylabel(r"$\phi$ / V @ $\phi_0=-0.5$ V ")
ax2.set_ylabel(r"$c_+$ / M @ $\phi_0=-0.5$ V ")
ax3.set_ylabel(r"$\varepsilon/\varepsilon_0$")
ax4.set_ylabel(r"$C$ / $\mu$F cm$^{-2}$")

ax1.set_ylim([PHI0_V, 0.05])
# ax2.set_ylim([0, 80])
ax3.set_ylim([0, 100])
# ax4.set_ylim([0, 150])

ax1.set_xlabel(r"$x$ / nm")
ax2.set_xlabel(r"$x$ / nm")
ax3.set_xlabel(r"$\phi_0$ / V")
ax4.set_xlabel(r"$\phi_0$ / V")

ax1.set_xlim([0, 5])
ax2.set_xlim([0, 5])
ax3.set_xlim([potentials[0], potentials[-1]])
ax4.set_xlim([potentials[0], potentials[-1]])

# ax1.set_xticks([0, 1, 2, 3, 4, 5])
# ax2.set_xticks([0, 1, 2, 3, 4, 5])
# ax3.set_xticks([0, 1, 2, 3, 4, 5])

ax1.legend(frameon=False)

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
plt.savefig("figures/intro-langevin-gouy-chapman-stern.pdf")

# fig2, ax2 = plt.subplots()
# for i, conc in enumerate(concentration_range):
#     ax2.plot(
#         x_axes[i],
#         cation_spatial[i],
#         label=f"{conc*1e3:.0f} mM",
#         color=colors1[i],
#     )
# ax2.set_xlim([0, 5])

plt.show()
