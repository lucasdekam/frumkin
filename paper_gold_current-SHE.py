"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from edl import models
from edl import constants as C
import plotting as P
import kinetics

DELTAG = 1.37 * C.E_0
DEFAULT_P_H_CATSIZE = 13
DEFAULT_P_H_CONCENT = 11
NUM_PTS = 200

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


potentials = np.linspace(-2, 0, NUM_PTS)


model = models.DoubleLayerModel(P.DEFAULT_CONC_M, P.DEFAULT_GAMMA, 2)
sol = model.potential_sweep(potentials)
current = kinetics.frumkin_corrected_current(
    sol,
    deltag=DELTAG,
)

fig = plt.figure(figsize=(3.25, 2.5))
ax3 = fig.add_subplot()


ax3.plot(
    potentials + C.AU_PZC_SHE_V,
    current * 1e-1,
    color="black",
)

ax3.set_xlim([-1.4, -0.8])
ax3.set_ylim([-1.5, 0.1])
ax3.set_xlabel(r"$\mathsf{E}$ / V vs. SHE")
ax3.set_ylabel(r"$j$ / mA cm$^{-2}$")
plt.tight_layout()

plt.savefig("figures/S1-gold-SHE-pH.pdf", dpi=240)

plt.show()
