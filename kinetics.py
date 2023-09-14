"""
Functions for converting PotentialSweepSolutions into reaction currents.
"""
import numpy as np

from edl import models
from edl import constants as C


def frumkin_corrected_current(
    model: models.DoubleLayerModel,
    potential_range_she: np.ndarray,
    pzc_she: float,
    p_h: float,
    deltag=1,
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
    sol = model.potential_sweep(potential_range_she - pzc_she, p_h=p_h)

    current = (
        -C.K_B
        * C.T
        / C.PLANCK
        * np.exp(-C.BETA * C.E_0 * deltag)
        * np.exp(-0.5 * C.E_0 * C.BETA * C.D_ADSORBATE_LAYER * sol["efield"].values)
        * C.C_WATER_BULK  # * sol["solvent"].values
    )
    return current


def marcus_current(
    model: models.DoubleLayerModel,
    potential_range_she: np.ndarray,
    pzc_she: float,
    p_h: float,
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
    sol = model.potential_sweep(potential_range_she - pzc_she, p_h=p_h)

    phi_rp = sol["phi0"].values - sol["efield"].values * C.D_ADSORBATE_LAYER
    # work = 6 * sol["pressure"].values / model.n_max
    delta_r_g = C.E_0 * (potential_range_she - phi_rp)

    e_act = (reorg + delta_r_g) ** 2 / (4 * reorg)
    print(np.min(e_act))

    current = -C.K_B * C.T / C.PLANCK * np.exp(-C.BETA * e_act) * C.C_WATER_BULK
    return current


def transport_limited_current(
    model: models.DoubleLayerModel,
    potential_range_she: np.ndarray,
    pzc_she: float,
    p_h: float,
    alpha: float,
    DELTAG=0.9 * C.E_0,
):
    """
    exp (- alpha beta [e0 phi + v dP])
    """
    sol = model.potential_sweep(potential_range_she - pzc_she, p_h=p_h)
    phi_rp = sol["phi0"].values - sol["efield"].values * C.D_ADSORBATE_LAYER
    # work = 1 * sol["pressure"].values / model.n_max

    current = (
        -C.K_B
        * C.T
        / C.PLANCK
        * np.exp(-alpha * C.BETA * (C.E_0 * phi_rp))
        * np.exp(-C.BETA * DELTAG)
    )
    return current
