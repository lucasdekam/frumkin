"""
Potential sweep tools for double-layer models
"""
from dataclasses import dataclass

import numpy as np
import constants as C

@dataclass
class PotentialSweepSolution:
    """
    Data class to store potential sweep solution data, e.g. differential capacitance
    phi: potential range, V
    charge: calculated surface charge, C/m^2
    cap: differential capacitance, uF/cm^2
    """
    phi:    np.ndarray
    charge: np.ndarray
    cap:    np.ndarray

def potential_sweep_guy_chapman(n_0, potential):
    """
    Analytic solution to a potential sweep in the Guy-Chapman model.
    """
    kappa_debye = np.sqrt(2*n_0*(C.Z*C.E_0)**2/(C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))

    cap = kappa_debye * C.EPS_R_WATER * C.EPS_0 * \
        np.cosh(C.BETA*C.Z*C.E_0 * potential / 2) * 1e2
    chg = np.sqrt(8 * n_0 * C.K_B * C.T * C.EPS_R_WATER * C.EPS_0) * \
        np.sinh(C.BETA * C.Z * C.E_0 * potential / 2)
    ret = PotentialSweepSolution(phi=potential, charge=chg, cap=cap)
    return ret
