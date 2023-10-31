"""
Making Gouy-Chapman-Stern theory plots for introduction
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import rcParams

# from matplotlib.gridspec import GridSpec
from PIL import Image

from edl import models
import plotting

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


fig = plt.figure(figsize=(5, 2.25))

ax3 = fig.add_subplot()


img = np.asarray(Image.open("figures/Climent2002_entropy.png"))
ax3.imshow(img)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.spines["top"].set_visible(False)
ax3.spines["bottom"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_xlabel(r"$\sigma$ / $\mu$C cm$^{-2}$")
ax3.set_ylabel(r"$\bar{s}$ / a.u.")
ax3.xaxis.set_label_position("top")


# labels = ["(a)", "(b)", "(c)"]
# for label, axis in zip(labels, fig.axes):
#     # label physical distance to the left and up:
#     trans = mtransforms.ScaledTranslation(-25 / 72, 10 / 72, fig.dpi_scale_trans)
#     axis.text(
#         0.0,
#         1.0,
#         label,
#         transform=axis.transAxes + trans,
#         fontsize="medium",
#         va="bottom",
#     )


plt.tight_layout()
plt.savefig("figures/intro-entropy.pdf", dpi=240)
plt.show()
