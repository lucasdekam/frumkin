"""
Make scatter plot with log j for various electrolytes
against phi0 - phi'
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

import plotting as P
import edl.constants as C

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

SPECIES = ["Li", "Na", "K"]
PH_RANGE = [11, 13]
PH_FILLSTYLE = {11: "none", 13: "full"}
SPECIES_COLORS = {"Li": "blue", "Na": "green", "K": "red"}


def plot_single_species(ax, species, dataframe):
    """
    Make scatter plot for a single species
    """
    dataframe = dataframe[dataframe["species"] == species]
    ph_dfs = {ph: dataframe.groupby("pH").get_group(ph) for ph in PH_RANGE}

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
                marker="o",
                fillstyle=PH_FILLSTYLE[ph],
                markeredgewidth=0.5,
                markersize=3,
                linestyle="None",
                color=colors[c],
                label=f"{c*1e3:.0f}",
            )


all_df = pd.read_csv("data/au_df_ringe.csv")
# all_df_ringe = pd.read_csv("data/all_df_ringe.csv")

## MAKE FIRST FIGURE
fig = plt.figure(figsize=(5.4167, 3))
ax_all = fig.add_subplot()

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

plot_single_species(ax_all, "Li", all_df)
plot_single_species(ax_all, "Na", all_df)
plot_single_species(ax_all, "K", all_df)

ax_all.set_xlim([-1.1, -0.6])
ax_all.set_ylim([-5, 0.5])


# labels = [r"(a) Li$^+$", r"(b) Na$^+$", r"(c) K$^+$", "(d)", "(e)", "(f)"]
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
ax_all.set_xlabel(r"$\phi_0 - \phi'$ / V")
ax_all.set_ylabel(r"$\log |j|$ / A cm$^{-2}$")

plt.tight_layout()
plt.subplots_adjust(left=0.3, right=0.8)
plt.savefig("figures/quantitative_ringe.pdf")
plt.show()
