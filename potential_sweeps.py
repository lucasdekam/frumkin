"""
Potential sweep tools for double-layer models
"""
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np

import constants as C
import models as M
import boundary_conditions as B
import spatial_profiles as S

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
    name:   str

def gouy_chapman(ion_concentration_molar: float, potential: np.ndarray):
    """
    Analytic solution to a potential sweep in the Gouy-Chapman model.

    ion_concentration_molar: bulk ion concentration in molar
    potential: potential array in V
    """
    n_0 = ion_concentration_molar*1e3*C.N_A
    kappa_debye = np.sqrt(2*n_0*(C.Z*C.E_0)**2/(C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))

    cap = kappa_debye * C.EPS_R_WATER * C.EPS_0 * \
        np.cosh(C.BETA*C.Z*C.E_0 * potential / 2) * 1e2
    chg = np.sqrt(8 * n_0 * C.K_B * C.T * C.EPS_R_WATER * C.EPS_0) * \
        np.sinh(C.BETA * C.Z * C.E_0 * potential / 2)
    return PotentialSweepSolution(phi=potential, charge=chg, cap=cap, name='Gouy-Chapman analytic')


def borukhov(ion_concentration_molar: float, a_m: float, potential: np.ndarray):
    """
    Analytic solution to a potential sweep in the Borukhov-Andelman-Orland model.

    ion_concentration_molar: bulk ion concentration in molar
    a_m: ion diameter in m
    potential: potential array in V
    """
    n_0 = ion_concentration_molar*1e3*C.N_A
    kappa_debye = np.sqrt(2*n_0*(C.Z*C.E_0)**2/(C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))
    chi_0 = 2 * a_m ** 3 * n_0

    y_0 = C.BETA*C.Z*C.E_0*potential  # dimensionless potential
    chg = np.sqrt(4 * C.K_B * C.T * C.EPS_0 * C.EPS_R_WATER * n_0 / chi_0) \
        * np.sqrt(np.log(chi_0 * np.cosh(y_0) - chi_0 + 1)) * y_0 / np.abs(y_0)
    cap = np.sqrt(2) * kappa_debye * C.EPS_R_WATER * C.EPS_0 / np.sqrt(chi_0) \
    * chi_0 * np.sinh(np.abs(y_0)) \
    / (chi_0 * np.cosh(y_0) - chi_0 + 1) \
    / (2*np.sqrt(np.log(chi_0 * np.cosh(y_0) - chi_0 + 1))) \
    * 1e2 # uF/cm^2
    return PotentialSweepSolution(phi=potential, charge=chg, cap=cap, name='Borukhov analytic')


def compute_charge(
        model: M.DoubleLayerModel,
        boundary_condition: B.BoundaryConditions,
        force_recalculation: bool=False):
    """
    Solve a model for a certain boundary condition with a default x-axis, and
    compute the surface charge.
    """
    x_axis_nm = S.get_x_axis_nm(100, 10000)
    sol = model.solve(x_axis_nm, boundary_condition, force_recalculation=force_recalculation)
    return sol.efield[0] * C.EPS_0 * sol.eps[0]


def numerical(
        model: M.DoubleLayerModel,
        potential: np.ndarray,
        force_recalculation: bool=False):
    """
    Numerical solution to a potential sweep for a defined double-layer model.
    """
    boundary_conditions = [B.Dirichlet(p) for p in potential]

    with mp.Pool(mp.cpu_count()) as pool:
        chg = pool.starmap(
            compute_charge,
            [(model, bc, force_recalculation) for bc in boundary_conditions])
        pool.close()

    cap = np.gradient(chg, edge_order=2)/np.gradient(potential) * 1e2
    return PotentialSweepSolution(phi=potential, charge=chg, cap=cap, name=model.name)
