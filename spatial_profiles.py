"""
Double-layer models
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class SpatialProfilesSolution:
    """
    Dataclass to store double-layer model solution data
    x: position, nm
    phi: potential, V
    efield: electric field, V/m
    c_dict: dict of concentration profiles of form {'name of species': np.ndarray}
    eps: relative permittivity
    name: name of the model
    """
    x:       np.ndarray
    phi:     np.ndarray
    efield:  np.ndarray
    c_dict:  dict
    eps:     np.ndarray
    name:    str


def get_x_axis_nm(xmax_nm, n_points):
    """
    Get a logarithmically spaced x-axis, fine mesh close to electrode
    """
    expmax = np.log10(xmax_nm)
    x_nm = np.logspace(-9, expmax, n_points) - 1e-9
    return x_nm


#     def computeCharge(self, xmax_m, N, potential_V, force_recalculation=False):
#         """
#         Compute the surface charge at a given potential
#         """
#         sol = self.getOdeSol(xmax_m, N, potential_V, verbose=False, force_recalculation=force_recalculation)

#         BFc, BFa, _, Omega = self.computeBoltzmannFactorsAndOmega(sol.y)
#         eps = self.computePermittivity(BFa, BFc, Omega, sol.y[1, :])
#         dphidx = - sol.y[1, :] * self.kappa / (C.beta * C.z * C.e_0)

#         charge_C = C.eps_0 * eps[0] * dphidx[0]
#         return charge_C

#     def getPotentialSweepSolution(self, potential_V, force_recalculation=False):
#         """
#         Compute the differential capacitance in microfarads per square cm.
#         Parallelized using the multiprocessing Python module.
#         """
#         xmax_m = 100e-9
#         N = 10000

#         pool = mp.Pool(mp.cpu_count())
#         charge = pool.starmap(self.computeCharge, [(xmax_m, N, phi, force_recalculation) for phi in potential_V])
#         pool.close()

#         cap = np.gradient(charge, edge_order=2) / np.gradient(potential_V) * 1e2
#         ret = PotentialSweepSolution(phi=potential_V, charge=charge, cap=cap)
#         return ret
