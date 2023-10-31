"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl import models

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

potentials = np.linspace(-1, 1, 200)

model = models.AqueousVariableStern(100e-3, 6, 6, 6, 6)
sweep = model.potential_sweep(potentials, tol=1e-3)

fig = plt.figure(figsize=(5, 2))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(
    sweep["phi0"],
    sweep["charge"] * 100,
    label="Cont.",
    color="black",
)
ax2.plot(sweep["phi0"], sweep["capacity"] * 100, color="black")

comsol_neg = np.loadtxt("comsol_data/70nm_nano_electrode_charge_neg.txt")
comsol_pos = np.loadtxt("comsol_data/70nm_nano_electrode_charge_pos.txt")
chg_neg = comsol_neg[:, 1] / model.kappa_debye**2 / (35e-9) ** 2 / np.pi
chg_pos = (
    (comsol_pos[:, 1] - comsol_pos[0, 1] + comsol_neg[0, 1])
    / model.kappa_debye**2
    / (35e-9) ** 2
    / np.pi
)
phi = np.concatenate([comsol_neg[::-1, 0], comsol_pos[:, 0]], axis=0)
chg = np.concatenate([chg_neg[::-1], chg_pos], axis=0)
cap = np.gradient(chg_neg[::-1], comsol_neg[::-1, 0])

ax1.plot(phi, chg, "k--", label="70nm")
ax2.plot(comsol_neg[::-1, 0], cap, "k--")
ax1.set_xlim([-1, 1])
ax1.set_ylabel(r"$\sigma$ / $\mu$C cm$^{-2}$")
ax1.set_xlabel(r"$\phi_0$ / V")
ax2.set_xlabel(r"$\phi_0$ / V")
ax1.legend(frameon=False)
ax2.set_ylabel(r"$C$ / $\mu$F cm$^{-2}$")
ax2.set_xlim([-1, 0])
ax2.set_ylim([0, 100])

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
plt.savefig("figures/comsol-cap.pdf")
plt.show()
