"""
Modelling the double layer using the Gongadze-Iglic approach.
"""

from typing import Optional, Dict, Literal
import numpy as np
from numpy import newaxis
from scipy import constants

from .solve.bvpsweep import sweep_solve_bvp

from .tools import langevin as L
from .tools.mesh import get_default_mesh
from .electrolyte import LatticeElectrolyte
from .results import VoltammetryResult, SinglePointResult
from . import boundary_conditions as bc


class GongadzeIglic:
    """
    Gongadze-Iglič double-layer model.
    """

    def __init__(
        self,
        electrolyte: LatticeElectrolyte,
        temperature: float = 298,
        xmax: float = 1000,
        **kwargs,
    ) -> None:

        self.el = electrolyte

        # Precompute thermal constants
        kbt = constants.Boltzmann * temperature
        self.kappa = (
            constants.elementary_charge**2
            / constants.epsilon_0
            / constants.angstrom
            / kbt
        )
        self.kbt_ev = kbt / constants.elementary_charge
        self.waterlayer_kwargs = kwargs

        # Mesh
        self.x_mesh = get_default_mesh("semi-infinite", xmax)

        # Pre-expand species properties for broadcasting
        self._q = self.el.ion_q[:, None]
        self._p = self.el.sol_p[:, None]
        self._ion_f_b = self.el.ion_f_b[:, None]
        self._sol_f_b = self.el.sol_f_b[:, None]
        self._ion_nb = self.el.ion_n_b[:, None]
        self._sol_nb = self.el.sol_n_b[:, None]
        self._ion_sizes = self.el.ion_sizes[:, None]
        self._sol_sizes = self.el.sol_sizes[:, None]

    # ------------------------------------------------------------------
    # Core physical helpers
    # ------------------------------------------------------------------

    def densities(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Ion and solvent densities from dimensionless potential y = (phi, phi').
        """
        phi, dphi = y
        ion_b = np.exp(-self._q * phi)
        sol_b = L.sinh_x_over_x(self._p * dphi)

        denom = (self._ion_f_b * ion_b).sum(0) + (self._sol_f_b * sol_b).sum(0)

        return (
            self._ion_nb * ion_b / denom,
            self._sol_nb * sol_b / denom,
        )

    def permittivity(self, y: np.ndarray) -> np.ndarray:
        """
        Relative permittivity from GI model.
        """
        _, sol_d = self.densities(y)
        return self.el.min_eps + self.kappa * (
            (self._p**2 * sol_d * L.langevin_x_over_x(self._p * y[1])).sum(0)
        )

    # ------------------------------------------------------------------
    # ODE
    # ------------------------------------------------------------------

    def ode_rhs(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        dy/dx = (phi', phi'').
        """
        phi, dphi = y
        dy0 = dphi

        ion_d, sol_d = self.densities(y)

        # Volume fractions
        ion_f = self._ion_sizes * ion_d / self.el.n_site
        sol_f = self._sol_sizes * sol_d / self.el.n_site

        # Charge and polarization terms
        f_1 = (self._q * ion_d).sum(0)
        Lpy = L.langevin_x(self._p * dphi)
        f_2 = ((self._p * dphi) * sol_d * Lpy * (self._q * ion_f).sum(0)).sum(0)

        g_1 = (self._p**2 * sol_d * L.d_langevin_x(self._p * dphi)).sum(0)
        g_2 = (self._p**2 * sol_d * Lpy**2 * (1 - sol_f.sum(0))).sum(0)

        denom = self.el.min_eps + self.kappa * (g_1 + g_2)

        dy1 = -(self.kappa * (f_1 + f_2)) / denom
        return np.vstack([dy0, dy1])

    # ------------------------------------------------------------------
    # Boundary Conditions
    # ------------------------------------------------------------------

    def boundary_condition(
        self, ya: np.ndarray, yb: np.ndarray, y0: float
    ) -> np.ndarray:
        """
        Stern layer / GI boundary condition wrapper.
        """
        eps = self.permittivity(ya.reshape(2, 1)).squeeze()
        return bc.waterlayer(ya, yb, y0, eps_ohp=eps, **self.waterlayer_kwargs)

    # ------------------------------------------------------------------
    # Voltammetry
    # ------------------------------------------------------------------

    def voltammetry(
        self, potential: np.ndarray, tol: float = 1e-3
    ) -> VoltammetryResult:
        """
        Solve BVP sweep and compute surface charge and capacitance.
        """
        y0 = np.zeros((2, len(self.x_mesh)))

        y = sweep_solve_bvp(
            fun=self.ode_rhs,
            bc=self.boundary_condition,
            x0=self.x_mesh,
            y0=y0,
            sweep_par=potential / self.kbt_ev,
            sweep_par_start=0.0,
            tol=tol,
        )

        dydx_surface = y[1, :, 0]
        perm = self.permittivity(y[:, :, 0])

        electric_field = -dydx_surface * self.kbt_ev / constants.angstrom
        surface_charge = constants.epsilon_0 * perm * electric_field * 100
        capacitance = np.gradient(surface_charge) / np.gradient(potential)

        return VoltammetryResult(
            potential=potential,
            surface_charge=surface_charge,
            capacitance=capacitance,
        )
