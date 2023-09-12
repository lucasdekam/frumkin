"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms
from matplotlib.gridspec import GridSpec
from PIL import Image

from edl import models
from edl import constants as C

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


DEFAULT_CONCENTRATION_M = 500e-3

ph_range = np.linspace(2, 13, 100)

gamma_list = [2, 4, 6, 8]
sol_list = []

for gamma in gamma_list:
    model = models.AqueousVariableStern(DEFAULT_CONCENTRATION_M, gamma, 2, 4, 1)
    sol = model.ph_sweep(ph_range, tol=1e-4)
    sol_list.append(sol)

fig = plt.figure(figsize=(5, 2.5))
gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])

ax_fsioh = fig.add_subplot(gs[0])
ax_img = fig.add_subplot(gs[1])

colors = [
    "tab:blue",
    "tab:red",
    "tab:green",
    "black",
]  # plotting.get_color_gradient(len(gamma_list))

for i, gamma in enumerate(gamma_list):
    frac_deprotonated = sol_list[i]["charge"].values / C.E_0
    ax_fsioh.plot(
        ph_range,
        frac_deprotonated / frac_deprotonated[-1],
        color=colors[i],
        label=f"{gamma:.0f}",
    )

# ax_fsioh.set_ylim([0.6, 1.0])
ax_fsioh.set_ylabel(r"Normalized $f_\mathrm{SiO^-}$")
# ax_fsioh.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax_fsioh.set_xticks([3, 5, 7, 9, 11, 13])
ax_fsioh.set_xlim([ph_range[0], ph_range[-1]])

img = np.asarray(Image.open("figures/ShafiulAzam.png"))
ax_img.imshow(img)
ax_img.set_xticks([])
ax_img.set_yticks([])
ax_img.spines["top"].set_visible(False)
ax_img.spines["bottom"].set_visible(False)
ax_img.spines["left"].set_visible(False)
ax_img.spines["right"].set_visible(False)
ax_img.set_ylabel(r"Normalized $f_\mathrm{SiO^-}$")

ax_fsioh.legend(frameon=False, title=r"$\gamma_+$")

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
    axis.set_xlabel(r"pH")

plt.tight_layout()

plt.savefig("figures/res-shafiul-azam.pdf", dpi=240)

plt.show()
