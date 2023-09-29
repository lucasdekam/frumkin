"""
silica
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

import plotting as P

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


r = np.linspace(1e-3, 1.2, 100)
a = 1
kr = 3.33
kc = 12.5
lb = 0.7

yuk = a * np.exp(-kr * r) / r
osc = a * np.exp(-kr * r) * np.cos(kc * r) / r
coulomb = lb / r


fig = plt.figure(figsize=(3, 2))
ax = fig.add_subplot()

color_blu = P.get_color_gradient(3)[1]
color_red = P.get_color_gradient(3, color="red")[1]

ax.plot(r, yuk, color=color_red, label="Yukawa")
ax.plot(r, osc, color=color_blu, label="Osc. Yukawa")
ax.set_ylim([-3, 9])
ax.set_xlim([0, 1.2])

ax.set_ylabel(r"$\beta u$")
ax.set_xlabel(r"$r$ / nm")

ax.legend(frameon=False)

plt.tight_layout()

plt.savefig("figures/yukawa.pdf")
plt.show()
