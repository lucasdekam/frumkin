"""
Implementation of double-layer models
"""

import numpy as np
from numpy import newaxis

from bvpsweep import sweep_solve_bvp

from . import langevin as L
from .electrolyte import LatticeElectrolyte
from scipy import constants


# KAPPA = constants.elementary_charge**2 / constants.epsilon_0 / constants.angstrom


class GongadzeIglic:
    """
    TODO: write
    """

    def __init__(
        self,
        electrolyte: LatticeElectrolyte,
        ohp: float,
        temperature: float = 298,
    ) -> None:
        self.el = electrolyte
        self.ohp = ohp
        kbt = constants.Boltzmann * temperature
        self.kappa = (
            constants.elementary_charge**2
            / constants.epsilon_0
            / constants.angstrom
            / kbt
        )
        self.kbt_ev = kbt / constants.elementary_charge

    def densities(self, y: np.ndarray):
        """
        Compute density profiles of all species
        """
        # Ion boltzmann factor
        ion_bf = np.exp(-self.el.ion_q[:, newaxis] * y[0, :])

        # Solvent boltzmann factor
        sol_bf = L.sinh_x_over_x(self.el.sol_p[:, newaxis] * y[1, :])

        # Denominator
        denom = np.sum(self.el.ion_f_b[:, newaxis] * ion_bf, axis=0) + np.sum(
            self.el.sol_f_b[:, newaxis] * sol_bf, axis=0
        )

        ion_densities = self.el.ion_n_b[:, newaxis] * ion_bf / denom
        sol_densities = self.el.sol_n_b[:, newaxis] * sol_bf / denom

        return ion_densities, sol_densities

    def ode_rhs(self, x, y):  # pylint: disable=unused-argument
        """
        Right hand side of the differential equation to pass to ODE solver
        """
        dy0 = y[1, :]

        ion_densities, sol_densities = self.densities(y)
        # Occupied volume fractions for shorter notation later
        ion_f = self.el.ion_sizes[:, newaxis] * ion_densities / self.el.n_site
        sol_f = self.el.sol_sizes[:, newaxis] * sol_densities / self.el.n_site

        f_1 = np.sum(self.el.ion_q[:, newaxis] * ion_densities, axis=0)
        f_2 = np.sum(
            self.el.sol_p[:, newaxis]
            * y[1, :]
            * sol_densities
            * L.langevin_x(self.el.sol_p[:, newaxis] * y[1, :])
            * np.sum(self.el.ion_q[:, newaxis] * ion_f, axis=0),
            axis=0,
        )
        g_1 = np.sum(
            self.el.sol_p[:, newaxis] ** 2
            * sol_densities
            * L.d_langevin_x(self.el.sol_p[:, newaxis] * y[1, :]),
            axis=0,
        )
        g_2 = np.sum(
            self.el.sol_p[:, newaxis] ** 2
            * sol_densities
            * L.langevin_x(self.el.sol_p[:, newaxis] * y[1, :]) ** 2
            * (1 - np.sum(sol_f, axis=0)),
            axis=0,
        )

        dy1 = (
            -self.kappa
            * (f_1 + f_2)
            / (self.el.min_eps + self.kappa * g_1 + self.kappa * g_2)
        )
        # print(self.kappa * f_1, self.kappa * f_2, self.kappa * g_1, self.kappa * g_2)
        return np.vstack([dy0, dy1])

    def boundary_condition(self, ya, yb, y0):
        """
        Stern layer boundary condition
        """
        return np.array(
            [
                ya[0] - y0 - ya[1] * self.ohp,
                yb[0],
            ]
        )

    def permittivity(self, y: np.ndarray):
        """
        Compute the permittivity
        """
        _, sol_densities = self.densities(y)
        return self.el.min_eps + self.kappa * np.sum(
            self.el.sol_p**2
            * sol_densities
            * L.langevin_x_over_x(self.el.sol_p[:, newaxis] * y[1, :]),
            axis=0,
        )

    def voltammetry(
        self, x_mesh: np.ndarray, potential: np.ndarray, tol: float = 1e-3
    ) -> np.ndarray:
        """
        Numerical solution to a potential sweep for a defined double-layer model.
        """
        y0 = np.zeros((2, len(x_mesh)))

        y = sweep_solve_bvp(
            fun=self.ode_rhs,
            bc=self.boundary_condition,
            x0=x_mesh,
            y0=y0,
            sweep_par=potential / self.kbt_ev,
            sweep_par_start=0.0,
            tol=tol,
        )

        return (
            -constants.epsilon_0
            * self.permittivity(y[:, :, 0])
            * y[1, :, 0]
            * self.kbt_ev
            / constants.angstrom
        )
