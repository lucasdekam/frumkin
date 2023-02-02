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