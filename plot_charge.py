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

EFF_D_WATER_M = (C.C_WATER_BULK * 1e3 * C.N_A) ** (-1 / 3)
GAMMA = 6  # (A_M / EFF_D_WATER_M) ** 3
A_M = GAMMA ** (1 / 3) * EFF_D_WATER_M

potentials = np.linspace(-1, 1, 100)
concentration_range = plotting.CONC_LIST

gcs = models.GouyChapmanStern(plotting.DEFAULT_CONC_M, A_M / 2)
lpb = models.LangevinPoissonBoltzmann(plotting.DEFAULT_CONC_M, A_M / 2)
bik = models.Borukhov(plotting.DEFAULT_CONC_M, A_M)
gon = models.AqueousVariableStern(plotting.DEFAULT_CONC_M, GAMMA, 2, 4, 4)
sol_gcs = gcs.potential_sweep(potentials, tol=1e-4)
sol_lpb = lpb.potential_sweep(potentials, tol=1e-4)
sol_bik = bik.potential_sweep(potentials, tol=1e-4)
sol_gon = gon.potential_sweep(potentials, tol=1e-4)

gi_conc = []

for i, conc in enumerate(concentration_range):
    gon = models.AqueousVariableStern(conc, GAMMA, 2, 4, 4)
    solution = gon.potential_sweep(potentials, p_h=7, tol=1e-4)
    gi_conc.append(solution)

fig = plt.figure(figsize=(5, 3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(potentials, sol_gcs["charge"] * 100, "k--", label="GCS")
ax1.plot(potentials, sol_lpb["charge"] * 100, "k:", label="LPB")
ax1.plot(potentials, sol_bik["charge"] * 100, "k-.", label="Bikerman")
ax1.plot(potentials, sol_gon["charge"] * 100, "k", label="GI")
ax1.set_xlim([0, 5])
ax1.legend(frameon=False)

colors1 = plotting.get_color_gradient(len(concentration_range))

for i, conc in enumerate(concentration_range):
    ax2.plot(
        potentials,
        gi_conc[i]["charge"] * 100,
        color=colors1[i],
        label=f"{conc*1e3:.0f}",
    )
ax2.legend(frameon=False, title=r"$c_+^*$ / mM")

labels = ["(a)", "(b)", "(c)", "(d)"]
for label, axis in zip(labels, fig.axes):
    trans = mtransforms.ScaledTranslation(-25 / 72, 10 / 72, fig.dpi_scale_trans)
    axis.text(
        0.0,
        1.0,
        label,
        transform=axis.transAxes + trans,
        fontsize="medium",
        va="bottom",
    )
    axis.set_ylim([-40, 40])
    axis.set_xlim([potentials[0], potentials[-1]])
    axis.set_ylabel(r"$\sigma$ / $\mu$C cm$^{-2}$")
    axis.set_xlabel(r"$\phi_0$ / V")
ax1.set_ylim([-100, 100])

plt.tight_layout()
plt.savefig("figures/res-charge-comparison.pdf")
plt.show()
