"""
Functions for converting PotentialSweepSolutions into reaction currents.
"""
import numpy as np
import edl
import constants as C

def frumkin_corrected_current(model: edl.DoubleLayerModel,
                            potential_range_she: np.ndarray,
                            pzc_she: float,
                            p_h: float) -> np.ndarray:
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

    current = - np.exp(-0.5*C.E_0*C.BETA * C.D_ADSORBATE_LAYER * sol['efield'].values) * sol['solvent'].values
    return current

def edl_transport_limited_current(model: edl.DoubleLayerModel,
                                   potential_range_she: np.ndarray,
                                   pzc_she: float,
                                   p_h: float) -> np.ndarray:
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

    current = -np.exp(-1.5 * C.E_0*C.BETA * sol['phi2'].values)
    return current
