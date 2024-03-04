"""
Make scatter plot with log j for various electrolytes
against phi0 - phi'
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms
from matplotlib.legend_handler import HandlerTuple

import plotting as P
import edl.constants as C


rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

SPECIES = ["Li", "K"]
PH_RANGE = [9, 10, 11, 12, 13]
PH_FILLSTYLE = {9: "full", 10: "full", 11: "none", 12: "full", 13: "full"}
PH_MARKERS = {9: "s", 10: "v", 11: "o", 12: "*", 13: "o"}
SPECIES_COLORS = {"Li": "blue", "K": "red"}


def plot_single_species(ax, species, df):
    """
    Make scatter plot for a single species
    """
    dataframe = df[df["species"] == species]
    ph_dfs = {
        ph: dataframe.groupby("pH").get_group(ph)
        for ph in PH_RANGE
        if ph in dataframe["pH"].values
    }

    unique_phs = []
    unique_concentrations = []

    for ph, df in ph_dfs.items():
        groupby_dict = df.groupby("c").indices
        colors = {
            list(groupby_dict.keys())[i]: color
            for i, color in enumerate(
                P.get_color_gradient(len(groupby_dict), color=SPECIES_COLORS[species])
            )
        }
        for c, indices in groupby_dict.items():
            select_df = df.iloc[indices]
            ax.plot(
                select_df["phi0"] - select_df["phi_rp"],
                select_df["log j"],
                marker=PH_MARKERS[ph],
                fillstyle=PH_FILLSTYLE[ph],
                markeredgewidth=0.5,
                markersize=3,
                linestyle="None",
                color=colors[c],
                # label=f"{c*1e3:.0f}",
            )

            if ph not in unique_phs:
                unique_phs.append(ph)
            if c not in unique_concentrations:
                unique_concentrations.append(c)

    onecolor = "black"
    ph_handles = []
    for ph in unique_phs:
        (p,) = ax.plot(
            [],
            [],
            color=onecolor,
            marker=PH_MARKERS[ph],
            fillstyle=PH_FILLSTYLE[ph],
            markeredgewidth=0.5,
            markersize=3,
            linestyle="None",
        )
        ph_handles.append(p)
    ph_legend = ax.legend(
        ph_handles,
        unique_phs,
        frameon=False,
        loc="lower right",
        title="pH",
    )

    conc_handles = []
    colors = P.get_color_gradient(
        len(unique_concentrations),
        color=SPECIES_COLORS[species],
    )
    for i, c in enumerate(unique_concentrations):
        (p,) = ax.plot(
            [],
            [],
            color=colors[i],
            linewidth=3,
        )
        conc_handles.append(p)
    conc_legend = ax.legend(
        conc_handles,
        [f"{c*1e3:.0f}" for c in unique_concentrations],
        frameon=False,
        loc="upper right",
        title=r"$c_+^*$ / mM",
        handlelength=1,
    )

    ax.add_artist(ph_legend)
    ax.add_artist(conc_legend)


all_df = pd.read_csv("data/pt_df.csv")
# all_df_ringe = pd.read_csv("data/all_df_ringe.csv")

## MAKE FIRST FIGURE
fig = plt.figure(figsize=(5.4167, 3))
ax_li = fig.add_subplot(121)
ax_k = fig.add_subplot(122)

# Fit Li data
li_select = all_df[all_df["species"] == "Li"]
li_select = li_select[li_select["pH"] == 13]
p = np.polyfit(
    C.BETA * C.E_0 * (li_select["phi0"] - li_select["phi_rp"]),
    li_select["log j"],
    deg=1,
)
print(p[0] * np.log(10))
phi_axis = np.linspace(-2, 0, 100)
logj_axis = np.polyval(p, C.BETA * C.E_0 * phi_axis)

# Plot on all axes and set labels
for axis in fig.axes:
    axis.plot(phi_axis, logj_axis, "k-")

plot_single_species(ax_li, "Li", all_df)
plot_single_species(ax_k, "K", all_df)

ax_li.set_xlim([-1.1, -0.4])
ax_k.set_xlim([-1.1, -0.4])
ax_li.set_ylim([-5, -1])
ax_k.set_ylim([-5, -1])

labels = [r"(a) Li$^+$", r"(b) K$^+$", "(d)", "(e)", "(f)"]
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
    axis.set_xlabel(r"$\phi_0 - \phi'$ / V")
    axis.set_ylabel(r"$\log |j|$ / A cm$^{-2}$")

plt.tight_layout()
plt.savefig("figures/quantitativept.pdf")
plt.show()
