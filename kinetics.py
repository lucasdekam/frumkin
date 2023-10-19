"""
Functions for converting PotentialSweepSolutions into reaction currents.
"""
import numpy as np
import pandas as pd

from edl import constants as C


def frumkin_corrected_current(
    sol: pd.DataFrame,
    deltag: float,
) -> np.ndarray:
    """
    Compute the Frumkin-corrected current:
    j ~ [H2O](x2) exp(-alpha * f * |E(D_ADSORBATE_LAYER)| * D_ADSORBATE_LAYER)
    (f=F/RT=beta*e0; assume alpha=0.5)

    We take the current to be negative (cathodic; reduction reaction)

    model: double layer model
    potential_range_she: potential range in V vs. SHE
    pzc_she: potential of zero charge of the metal in V vs. SHE
    """
    current = (
        -2
        * C.E_0
        / C.BETA
        / C.PLANCK
        * np.exp(-C.BETA * deltag)
        * np.exp(-0.5 * C.E_0 * C.BETA * (sol["phi0"] - sol["phi_rp"]))
        * 5e18
    )
    return current
    # return sol["phi0"] - sol["phi_rp"]


def marcus_current(
    sol: pd.DataFrame,
    pzc_she: float,
    reorg: float,
) -> np.ndarray:
    """
    Compute the current that is rate-limited by transport away from the
    electrode surface:
    j ~ exp(-f * phi2)
    (f=F/RT=beta*e0)

    We take the current to be negative (cathodic; reduction reaction)

    model: double layer model
    potential_range_she: potential range in V vs. SHE
    pzc_she: potential of zero charge of the metal in V vs. SHE
    """
    delta_r_g = C.E_0 * (sol["phi0"] - sol["phi_rp"] + pzc_she)
    e_act = (reorg + delta_r_g) ** 2 / (4 * reorg)
    current = -2 * C.E_0 / C.BETA / C.PLANCK * np.exp(-C.BETA * e_act) * 5e18
    return current
    # return sol["phi0"] - sol["phi_rp"]


def transport_limited_current(
    sol: pd.DataFrame,
    alpha: float,
    deltag: float,
):
    """
    exp (- alpha beta [e0 phi + v dP])
    """
    # phi_rp = sol["phi0"].values - sol["efield"].values * C.D_ADSORBATE_LAYER
    # work = 1 * sol["pressure"].values / model.n_max

    current = (
        -2
        * C.E_0
        / C.BETA
        / C.PLANCK
        * 5e18
        * np.exp(-alpha * C.BETA * (C.E_0 * sol["phi_rp"]))
        * np.exp(-C.BETA * deltag)
    )
    return current
