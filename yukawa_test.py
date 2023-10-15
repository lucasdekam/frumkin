"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl import models
from edl import constants as C
import plotting as P

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

fig = plt.figure(figsize=(3,2.5))
ax = fig.add_subplot()

sizes_nm = [0.2, 0.4, 0.6, 0.8]
# charges = [-0.172, -0.173, -0.179, -0.188]

for lh in sizes_nm:
    yukawa = models.Yukawa(100e-3, lh * 1e-9, 1 / 0.3e-9)
    sol = yukawa.spatial_profiles(-C.E_0*1e18, tol=1e-3)
    ax.plot(sol['x'], sol['cations'], label=f'{lh:.1f}')

ax.set_xlim([0,4])
ax.set_ylim([0, 2])

ax.set_xlabel(r'$x$ / nm')
ax.set_ylabel(r'$c_+$ / M')
ax.legend(frameon=False, title=r'$l_\mathrm{h}$ / nm')
plt.tight_layout()
plt.savefig('figures/yukawa-cation-conc.pdf')
plt.show()
