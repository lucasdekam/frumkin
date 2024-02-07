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

    # I'm lazy so I predefine concentrations
    legend_entries = {
        0.005: [],
        0.025: [],
        0.050: [],
        0.100: [],
    }

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
            (p,) = ax.plot(
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

            legend_entries[c].append(p)

    print([tuple(legend_entries[c]) for c in legend_entries.keys()])
    print(list(legend_entries.keys()))
    ax.legend(
        [tuple(legend_entries[c]) for c in legend_entries.keys()],
        [f"{c*1e3:.0f}" for c in legend_entries.keys()],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
        title=r"$c^*$ / mM",
    )


all_df = pd.read_csv("data/pt_df.csv")
# all_df_ringe = pd.read_csv("data/all_df_ringe.csv")

## MAKE FIRST FIGURE
fig = plt.figure(figsize=(3.248, 5))
ax_li = fig.add_subplot(211)
ax_k = fig.add_subplot(212)

# Fit Li data
li_select = all_df[all_df["species"] == "Li"]
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
