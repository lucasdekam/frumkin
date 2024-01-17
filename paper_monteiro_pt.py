"""
Making an analysis of Monteiro's data
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

from edl.models import DoubleLayerModel, ExplicitStern
from edl import constants as C

rcParams["lines.linewidth"] = 0.75
rcParams["font.size"] = 8
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5


def process(edl_constructor, dataframe: pd.DataFrame, pzc_she: float):
    """
    Process Monteiro data and calculate phi0-phi'
    """
    dataframe["E_SHE"] = dataframe["E_RHE"] - dataframe["pH"] * 59e-3
    dataframe["phi0"] = dataframe["E_SHE"] - pzc_she

    # Find unique log concentration values
    unique_logc_values = dataframe.drop_duplicates(subset="log c")["log c"].values

    # One computation for each concentration value
    processed_dfs = []

    for logc in unique_logc_values:
        df_select_logc = dataframe[dataframe["log c"] == logc].copy()
        model = edl_constructor(10**logc)

        potentials_vs_pzc = np.linspace(np.min(df_select_logc["phi0"]), 0, 50)
        sweep = model.potential_sweep(potentials_vs_pzc, tol=1e-4)

        # Interpolate the result for the phi0 values already in the dataframe
        interpolator = interp1d(potentials_vs_pzc, sweep["phi_rp"])
        df_select_logc["phi_rp"] = interpolator(df_select_logc["phi0"])
        processed_dfs.append(df_select_logc)

    return pd.concat(processed_dfs)


# Preprocess dataframes
df_pt_li = pd.read_parquet("data/pt_li.parquet")
df_pt_li["name"] = [
    r"Li$^+$ pH 11" if value == 11 else r"Li$^+$ pH 13" for value in df_pt_li["pH"]
]
df_pt_li["color"] = "black"
df_pt_li["fillstyle"] = ["full" if value == 13 else "none" for value in df_pt_li["pH"]]
df_pt_li["marker"] = "o"

df_pt_k = pd.read_parquet("data/pt_k.parquet")
df_pt_k["name"] = [r"K$^+$ pH" + f" {value:0d}" for value in df_pt_k["pH"]]
df_pt_k["color"] = "red"
df_pt_k["fillstyle"] = [
    "none" if value in [9, 11] else "full" for value in df_pt_k["pH"]
]
df_pt_k["marker"] = df_pt_k["pH"]
df_pt_k["marker"].replace(
    to_replace={
        9: "v",
        10: "v",
        11: "o",
        12: "s",
        13: "o",
    },
    inplace=True,
)

# Calculate phi0 - phi' for normal model
df_pt_li_1 = process(
    lambda c: DoubleLayerModel(c, 7, 2),
    df_pt_li,
    C.AU_PZC_SHE_V,
)
df_pt_k_1 = process(
    lambda c: DoubleLayerModel(c, 5, 2),
    df_pt_k,
    C.AU_PZC_SHE_V,
)
final_df_1 = pd.concat([df_pt_li_1, df_pt_k_1])

# Calculate phi0 - phi' for larger Stern model
df_pt_li_2 = process(
    lambda c: ExplicitStern(c, 7, 2, 5.8e-10),
    df_pt_li,
    C.AU_PZC_SHE_V,
)
df_pt_k_2 = process(
    lambda c: ExplicitStern(c, 5, 2, 4.1e-10),
    df_pt_k,
    C.AU_PZC_SHE_V,
)
final_df_2 = pd.concat([df_pt_li_2, df_pt_k_2])

fig = plt.figure(figsize=(5, 3))
ax_au1 = fig.add_subplot(121)
ax_au2 = fig.add_subplot(122)

for df, ax in zip([final_df_1, final_df_2], [ax_au1, ax_au2]):
    row_names_seen = [
        r"Li$^+$ pH 11",
        r"Li$^+$ pH 13",
        r"K$^+$ pH 11",
        r"K$^+$ pH 13",
    ]
    for i, row in df.iterrows():
        label = None
        if row["name"] not in row_names_seen:
            label = row["name"]
            row_names_seen.append(label)
        ax.plot(
            row["phi0"] - row["phi_rp"],
            row["log j"],
            label=label,
            marker=row["marker"],
            fillstyle=row["fillstyle"],
            markersize=3,
            markeredgewidth=0.5,
            linestyle="None",
            color=row["color"],
        )


x = C.E_0 * C.BETA * (df_pt_li_1["phi0"] - df_pt_li_1["phi_rp"])
y = df_pt_li_1["log j"]
p1 = np.polyfit(x, y, deg=1)
print(p1)

x = C.E_0 * C.BETA * (df_pt_li_2["phi0"] - df_pt_li_2["phi_rp"])
y = df_pt_li_2["log j"]
p2 = np.polyfit(x, y, deg=1)
print(p2)

x_ax = np.linspace(-1.5, 0, 100)
ax_au1.plot(x_ax, np.polyval(p1, C.E_0 * C.BETA * x_ax), "k")
ax_au2.plot(x_ax, np.polyval(p2, C.E_0 * C.BETA * x_ax), "k")

ax_au1.set_xlim([-1.3, -0.8])
ax_au2.set_xticks([-1.3, -1.2, -1.1, -1.0, -0.9, -0.8])
ax_au2.set_xlim([-0.95, -0.45])
ax_au2.set_xticks([-0.9, -0.8, -0.7, -0.6, -0.5])
ax_au1.legend(frameon=False)

labels = [r"(a) $x_2$ in this work", r"(b) $x_2$ from Ringe et al."]
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
plt.savefig("figures/S3-res-monteiro-pt.pdf")

plt.show()
