"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec

from edl import models
from edl import constants as C
import plotting as P
import kinetics

DELTAG = 1.37 * C.E_0
DEFAULT_PH = 11
NUM_PTS = 200

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


potentials = np.linspace(-2, 0, NUM_PTS)

conc_list = [5e-3, 250e-3, 500e-3, 1000e-3]

current_conc_frumkin = np.zeros((len(conc_list), NUM_PTS))

for i, conc in enumerate(conc_list):
    model = models.DoubleLayerModel(conc, 5, 2)
    sol = model.potential_sweep(potentials)
    current_conc_frumkin[i, :] = kinetics.frumkin_corrected_current(
        sol,
        deltag=DELTAG,
        alpha=0.36,
    )

fig = plt.figure(figsize=(5.4167, 4))
ax1 = fig.add_subplot()

colors = P.get_color_gradient(len(conc_list))

for i, conc in enumerate(conc_list):
    ax1.plot(
        potentials + C.PT_PZC_SHE_V + 59e-3 * DEFAULT_PH,
        -np.gradient(potentials, np.log10(-current_conc_frumkin[i, :])) * 1e3,
        color=colors[i],
        label=f"{conc*1e3:.0f}",
    )

ax1.legend(loc="lower right", frameon=False, title=r"$c_+^*$ / mM")
ax1.set_xlabel(r"$\mathsf{E}$ vs. SHE / V")
ax1.set_ylabel(r"Tafel slope / mV")
ax1.set_ylim([150, 250])

plt.tight_layout()

plt.show()
