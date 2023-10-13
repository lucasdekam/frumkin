"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl import models
import plotting

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

phi0_v_range = [-0.4, -0.8, -1.2, -1.6, -2.0]

sols = []
sweeps = []

for i, phi0_v in enumerate(phi0_v_range):
    model = models.AqueousVariableStern(
        plotting.DEFAULT_CONC_M, plotting.DEFAULT_GAMMA, 2, 5, 1
    )
    spatial = model.spatial_profiles(phi0=phi0_v, tol=1e-3)
    sols.append(spatial)

# fig, ax = plt.subplots(figsize=(5, 4), nrows=2, ncols=2)
fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)

colors1 = plotting.get_color_gradient(len(phi0_v_range), color="purple")

for i, phi0_v in enumerate(phi0_v_range):
    ax1.plot(
        sols[i]["x"],
        sols[i]["phi"],
        label=f"{phi0_v:.1f}",
        color=colors1[i],
    )

    ax2.plot(
        sols[i]["x"],
        sols[i]["cations"],
        color=colors1[i],
    )

    ax3.plot(
        sols[i]["x"],
        sols[i]["eps"],
        color=colors1[i],
    )

ax1.set_ylabel(r"$\phi$ / V")
ax2.set_ylabel(r"$c_+$ / M")
ax3.set_ylabel(r"$\varepsilon/\varepsilon_0$")

ax1.set_ylim([np.min(phi0_v_range), 0.05])
ax2.set_ylim([0, 8])
ax3.set_ylim([0, 80])

ax1.set_xlabel(r"$x$ / nm")
ax2.set_xlabel(r"$x$ / nm")
ax3.set_xlabel(r"$x$ / nm")

ax1.set_xlim([0, 1])
ax2.set_xlim([0, 2])
ax3.set_xlim([0, 2])

# ax[0, 0].set_xticks([0, 1, 2, 3, 4, 5])

ax1.legend(frameon=False, title=r"$\phi_0$ / V")

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
plt.savefig("figures/intro-gongadze-iglic-phi.pdf")
plt.show()
