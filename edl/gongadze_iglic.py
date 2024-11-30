"""
Implementation of double-layer models
"""

from typing import Dict
import numpy as np

from bvpsweep import sweep_solve_bvp

from . import langevin as L


class GongadzeIglic:
    """
    TODO: write
    """

    def __init__(self, species: Dict, ohp: float, min_eps: float) -> None:
        self.species = species
        self.charges = np.array(
            [s["charge"] if "charge" in s else 0 for _, s in species.items()]
        ).reshape(-1, 1)
        self.sizes = np.array(
            [s["size"] if "size" in s else 1 for _, s in species.items()]
        ).reshape(-1, 1)
        self.fractions = np.array([s["fraction"] for _, s in species.items()]).reshape(
            -1, 1
        )
        self.dipoles = np.array(
            [s["dipole"] if "dipole" in s else 0 for _, s in species.items()]
        ).reshape(-1, 1)
        self.min_eps = min_eps
        self.ohp = ohp

        assert len(self.fractions) == len(
            species
        ), "Specify the occupied volume fraction ('fraction') of all species."
        total_occupied = np.sum(self.sizes * self.fractions)
        assert np.isclose(
            total_occupied, 1.0
        ), f"Fractions*sizes must add up to 1.0, now {total_occupied:.3f}."

        print("Number of species: %.0f" % len(species))
        print("Charges: %s" % str(self.charges.squeeze()))
        print("Sizes: %s" % str(self.sizes.squeeze()))
        print("Volume fractions: %s" % str(self.fractions.squeeze()))
        print("Dipoles: %s" % str(self.dipoles.squeeze()))

    def densities(self, y: np.ndarray):
        """
        Compute density profiles of all species (as dimensionless fraction of
        occupied volume)
        """
        bf = np.exp(-self.charges * y[0, :]) * L.sinh_x_over_x(self.dipoles * y[1, :])
        denom = np.sum(self.sizes * self.fractions * bf, axis=0)
        profile = self.fractions * bf / denom
        return profile

    def ode_rhs(self, x, y):  # pylint: disable=unused-argument
        """
        Function to pass to ODE solver, specifying the right-hand side of the
        system of dimensionless 1st order ODE's that we solve.
        x: dimensionless x-axis of length n, i.e. kappa (1/m) times x-position (m).
        y: dimensionless potential phi and dphi/dx, size 2 x n.
        """
        dy0 = y[1, :]
        densities = self.densities(y)

        f_1 = np.sum(-self.charges * densities, axis=0)
        f_2 = np.sum(
            -self.dipoles
            * densities
            * y[1, :]
            * L.langevin_x(self.dipoles * y[1, :])
            * np.sum(densities * self.charges * self.sizes, axis=0),
            axis=0,
        )
        g_1 = self.min_eps
        g_2 = np.sum(
            self.dipoles**2 * densities * L.d_langevin_x(self.dipoles * y[1, :]), axis=0
        )
        g_3 = np.sum(
            self.dipoles**2
            * densities
            * L.langevin_x(self.dipoles * y[1, :]) ** 2
            * np.sum(densities[self.charges.squeeze() > 0], axis=0),
            axis=0,
        )
        # (1 - np.sum(densities[self.dipoles.squeeze() > 0], axis=0))

        dy1 = (f_1 + f_2) / (g_1 + g_2 + g_3)
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

    def permittivity(self, sol_y: np.ndarray):
        """
        Compute the permittivity using the electric field
        n_sol: solvent number density
        y_1: dimensionless electric field
        """
        return self.min_eps + np.sum(
            self.dipoles**2
            * self.densities(sol_y)
            * L.langevin_x_over_x(self.dipoles * sol_y[1, :]),
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
            sweep_par=potential,
            sweep_par_start=0.0,
            tol=tol,
        )

        return -self.permittivity(y[:, :, 0]) * y[1, :, 0]
