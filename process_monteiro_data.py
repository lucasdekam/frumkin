import pandas as pd
import numpy as np
from edl.models import ExplicitStern
from scipy.interpolate import interp1d


def process(dataframe: pd.DataFrame, gamma: float, x_2: float, pzc_she: float):
    dataframe["E_SHE"] = dataframe["E_RHE"] - dataframe["pH"] * 59e-3
    dataframe["phi0"] = dataframe["E_SHE"] - pzc_she

    # Find unique log concentration values
    unique_logc_values = dataframe.drop_duplicates(subset="log c")["log c"].values

    # One computation for each concentration value
    processed_dfs = []

    for logc in unique_logc_values:
        df_select_logc = dataframe[dataframe["log c"] == logc].copy()
        model = ExplicitStern(10**logc, gamma, 2, x_2)

        potentials_vs_pzc = np.linspace(np.min(df_select_logc["phi0"]), 0, 50)
        sweep = model.potential_sweep(potentials_vs_pzc, tol=1e-4)

        # Interpolate the result for the phi0 values already in the dataframe
        interpolator = interp1d(potentials_vs_pzc, sweep["phi_rp"])
        df_select_logc["phi_rp"] = interpolator(df_select_logc["phi0"])
        processed_dfs.append(df_select_logc)

    return pd.concat(processed_dfs)
