"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl import constants as C
from edl import models
from edl import langevin as L 

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

model = models.AqueousVariableStern(1, 6, 6, 6, 6)

def permittivity(x):
    nw = C.C_WATER_BULK * 1e3 * C.N_A
    p = model.p_water
    return 1.33 ** 2 + C.BETA * nw * p ** 2 * L.langevin_x_over_x(x) / C.EPS_0


x = np.linspace(-25, 25, 100)

fig = plt.figure(figsize=(5, 2))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(x, L.langevin_x(x), color='black')
ax2.plot(x, 
         permittivity(x),
         color='black')

ax1.set_ylabel(r"$\mathcal{P} / n_\mathrm{w}p$")
ax2.set_ylabel(r"$\varepsilon / \varepsilon_0$")

ax1.set_ylim([-1, 1])
ax2.set_ylim([0, 80])

ax1.set_xlabel(r"$\beta p E$")
ax2.set_xlabel(r"$\beta p E$")

# ax1.legend(frameon=False, title=r"$c_0$ / mM")

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
    axis.set_xlim([x[0], x[-1]])

plt.tight_layout()
plt.savefig("figures/intro-pol-eps.pdf")

plt.show()
