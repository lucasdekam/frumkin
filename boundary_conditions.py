"""
Boundary condition tools for double-layer models
"""
from abc import ABC, abstractmethod
import numpy as np

import constants as C

class BoundaryConditions(ABC):
    """
    Abstract base class for boundary conditions. Makes sure that each class
    has a function to pass to an ODE solver, and a name property
    """
    @abstractmethod
    def func(self, ya, yb):
        """
        Function to pass to an ODE solver
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the boundary condition
        """

class Dirichlet(BoundaryConditions):
    """
    Dirichlet boundary condition for the potential.

    phi_0: potential at the electrode in V, referenced to the solution bulk
    """
    def __init__(self, phi_0) -> None:
        self.phi_0 = phi_0
        self.name = f"Dirichlet_{phi_0:.3f}V"

    def func(self, ya, yb):
        return np.array([ya[0] - C.BETA * C.Z * C.E_0 * self.phi_0, yb[0]])

    def get_name(self) -> str:
        return self.name


class Insulator(BoundaryConditions):
    """
    Robin boundary condition for an insulator.
    """
    def __init__(self, p_h: float, model) -> None:
        self.p_h = p_h
        self.c_proton_bulk = 10 ** (-p_h)
        self.n_proton_bulk = self.c_proton_bulk * 1e3 * C.N_A
        self.name = f"Insulator_pH{p_h:.0f}"

        self.kappa_debye = model.kappa_debye
        self.c_denom_f = model.density_denominator
        self.c_dict_f = model.concentration_dict

    def func(self, ya, yb):
        denom = self.c_denom_f(ya.reshape(2, 1))
        c_proton_surf = self.c_dict_f(ya.reshape(2, 1), denom)[r'H$^+$'].squeeze()

        left = C.EPS_R_WATER * C.EPS_0 * ya[1] * self.kappa_debye / (C.BETA * C.Z * C.E_0) \
            - C.E_0 * C.N_SITES_SILICA * \
            C.K_SILICA / (C.K_SILICA  + c_proton_surf)
        right = yb[0]
        return np.array([left, right])

    def get_name(self) -> str:
        return self.name