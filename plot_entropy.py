"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from PIL import Image

from edl import models
import plotting

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


def entropy(x):  # pylint: disable=invalid-name
    """
    Calculate the volumetric entropy density
    """
    s_over_nkb = 1 - x / np.tanh(x) + np.log(np.sinh(x) / x)
    return s_over_nkb


potentials = np.linspace(-0.8, 0.8, 200)
concentration = [0.01]

entropy_conc = np.zeros((len(concentration), len(potentials)))
charge = np.zeros((len(concentration), len(potentials)))

for i, conc in enumerate(concentration):
    gc = models.LangevinGouyChapmanStern(conc, eps_r_opt=1)
    gc_nsol = gc.potential_sweep(potentials, tol=1e-3)
    entropy_conc[i, :] = gc_nsol["entropy"]
    charge[i, :] = gc_nsol["charge"]

# fig, ax = plt.subplots(figsize=(5, 4), nrows=2, ncols=2)
fig = plt.figure(figsize=(5, 4))
gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 3])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

colors = plotting.get_color_gradient(len(concentration))

bpe = np.linspace(-4, 4, 100)
ax1.plot(
    bpe,
    entropy(bpe),
    color="black",
)

for i, conc in enumerate(concentration):
    ax2.plot(
        charge[i, :] * 100,
        entropy_conc[i, :],
        label=r"$c_0=$" + f"{conc*1e3:.0f} mM",
        color="black",  # colors[i],
    )

ax1.set_ylabel(r"$s/n_\mathrm{w}k_\mathrm{B}$")
ax2.set_ylabel(r"$s(0)/n_\mathrm{w}k_\mathrm{B}$")

ax1.set_xlabel(r"$\beta p E$")
ax2.set_xlabel(r"$\sigma$ / $\mu$C cm$^{-2}$")

ax1.set_xlim([bpe[0], bpe[-1]])
ax2.set_xlim([-50, 50])

ax1.set_ylim([-1.5, 0.2])
ax2.set_ylim([-1.5, 0.2])

ax2.legend(frameon=False)

img = np.asarray(Image.open("figures/Climent2002_entropy.png"))
ax3.imshow(img)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.spines["top"].set_visible(False)
ax3.spines["bottom"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_xlabel(r"$\sigma$ / $\mu$C/cm$^2$")
ax3.set_ylabel(r"$\bar{s}$ / a.u.")
ax3.xaxis.set_label_position("top")


labels = ["(a)", "(b)", "(c)"]
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
plt.savefig("figures/intro-entropy.pdf", dpi=240)
plt.show()
