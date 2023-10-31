"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from edl import models
from edl import constants as C
import plotting

rcParams["lines.linewidth"] = 1

DEFAULT_CONCENTRATION_M = 1e-3
potentials = np.linspace(-0.1, 0.1, 100)
temperature = 273 + np.arange(10, 90, 20)

sigma_temp = np.zeros((len(temperature), len(potentials)))

for i, temp in enumerate(temperature):
    gc = models.GouyChapmanStern(DEFAULT_CONCENTRATION_M)
    n_0 = DEFAULT_CONCENTRATION_M * C.N_A * 1e3
    gc.kappa_debye = np.sqrt(
        2 * n_0 * C.E_0**2 / (C.EPS_R_WATER * C.EPS_0 * C.K_B * temp)
    )
    gc_nsol = gc.potential_sweep(potentials, tol=1e-3)
    sigma_temp[i, :] = gc_nsol["charge"]


fig, ax = plt.subplots(figsize=(4, 3), nrows=1, ncols=1, sharex=True)
colors = plotting.get_color_gradient(len(temperature))

for i, temp in enumerate(temperature):
    ax.plot(
        potentials,
        sigma_temp[i, :] * 100,
        label=r"$T=$" + f"{temp - 273:.0f}" + r"$^\circ$C",
        color=colors[i],
    )


ax.set_ylabel(r"$\sigma$ [$\mu$C/cm$^2$]")
ax.set_xlabel(r"$\phi_\mathrm{M}$")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
