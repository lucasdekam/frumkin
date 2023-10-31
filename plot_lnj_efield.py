"""
Making volcano plots
"""
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl import constants as C

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

e_au, i_au, name_au = pickle.load(open(r"au_data.pickle", "rb"))
e_pt, i_pt, name_pt = pickle.load(open(r"pt_data.pickle", "rb"))


def plot(axis, efield, current, name, style_dict, dont_label=False):
    """
    plot stuff
    """
    if dont_label:
        axis.plot([e * 1e-9 for e in efield], current, **style_dict)
    else:
        axis.plot([e * 1e-9 for e in efield], current, label=name, **style_dict)


fig = plt.figure(figsize=(5, 3))
ax_au = fig.add_subplot(121)
ax_pt = fig.add_subplot(122)

drivingforce = np.linspace(-5, -3.7, 100) * 1e9
fit = np.polyfit(e_au[0] + e_au[1], i_au[0] + i_au[1], deg=1)
print(fit)
ax_au.plot(drivingforce * 1e-9, np.polyval(fit, drivingforce), "k")

drivingforce = np.linspace(-3.8, -2.3, 100) * 1e9
fit = np.polyfit(e_pt[0] + e_pt[1] + e_pt[2], i_pt[0] + i_pt[1] + i_pt[2], deg=1)
print(fit)
ax_pt.plot(drivingforce * 1e-9, np.polyval(fit, drivingforce), "k")

plot(
    ax_au,
    e_au[0],
    i_au[0],
    name_au[0],
    dict(
        marker="o",
        fillstyle="none",
        markersize=3,
        markeredgewidth=0.5,
        linestyle="None",
        color="black",
    ),
)

plot(
    ax_au,
    e_au[1],
    i_au[1],
    name_au[1],
    dict(
        marker="o",
        fillstyle="full",
        markersize=3,
        markeredgewidth=0.5,
        linestyle="None",
        color="black",
    ),
)

plot(
    ax_au,
    e_au[2],
    i_au[2],
    name_au[2],
    dict(
        marker="o",
        fillstyle="none",
        markersize=3,
        markeredgewidth=0.5,
        linestyle="None",
        color="red",
    ),
)

plot(
    ax_au,
    e_au[3],
    i_au[3],
    name_au[3],
    dict(
        marker="o",
        fillstyle="full",
        markersize=3,
        markeredgewidth=0.5,
        linestyle="None",
        color="red",
    ),
)

plot(
    ax_pt,
    e_pt[0],
    i_pt[0],
    name_pt[0],
    dict(
        marker="o",
        fillstyle="none",
        markersize=3,
        markeredgewidth=0.5,
        linestyle="None",
        color="black",
    ),
    dont_label=True,
)

plot(
    ax_pt,
    e_pt[1],
    i_pt[1],
    name_pt[1],
    dict(
        marker="o",
        fillstyle="full",
        markersize=3,
        markeredgewidth=0.5,
        linestyle="None",
        color="black",
    ),
    dont_label=True,
)

plot(
    ax_pt,
    e_pt[2],
    i_pt[2],
    name_pt[2],
    dict(
        marker="v",
        fillstyle="none",
        markersize=3,
        markeredgewidth=0.5,
        linestyle="None",
        color="red",
    ),
)

plot(
    ax_pt,
    e_pt[3],
    i_pt[3],
    name_pt[3],
    dict(
        marker="v",
        fillstyle="full",
        markersize=3,
        markeredgewidth=0.5,
        linestyle="None",
        color="red",
    ),
)

plot(
    ax_pt,
    e_pt[4],
    i_pt[4],
    name_pt[4],
    dict(
        marker="o",
        fillstyle="none",
        markersize=3,
        markeredgewidth=0.5,
        linestyle="None",
        color="red",
    ),
    dont_label=True,
)

plot(
    ax_pt,
    e_pt[5],
    i_pt[5],
    name_pt[5],
    dict(
        marker="s",
        fillstyle="full",
        markersize=3,
        markeredgewidth=0.5,
        linestyle="None",
        color="red",
    ),
)


ax_au.legend(frameon=False, fancybox=False)
ax_pt.legend(frameon=False, fancybox=False)

labels = ["(a) Au", "(b) Pt", "(c)", "(d)", "(e)", "(f)"]
for label, ax in zip(labels, fig.axes):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-25 / 72, 10 / 72, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize="medium",
        va="bottom",
    )
    ax.set_xlabel(r"$\phi_0 - \phi'$ / V")
    ax.set_ylabel(r"log $|j|$ / A cm$^{-2}$")
    ax.set_ylim([-5, 0.5])

plt.tight_layout()
plt.savefig("figures/res-lnj-efield.pdf")

plt.show()
