"""
Modelling the double layer using the Gongadze-Iglic approach.
"""

from typing import Optional, Dict
import numpy as np
from numpy import newaxis
from scipy import constants

from .solve.bvpsweep import sweep_solve_bvp

from .tools import langevin as L
from .tools.mesh import get_default_mesh
from .electrolyte import LatticeElectrolyte
from .results import VoltammetryResult, SinglePointResult
from .boundary import Boundary


class GongadzeIglic:
    """
    A class to model the electric double layer with the approach of Gongadze and Iglic.

    Parameters
    ----------
    electrolyte : LatticeElectrolyte
        The electrolyte model.
    boundary : Boundary
        Boundary condition (SemiInfinite, Symmetric, or Antisymmetric).
    temperature : float, optional
        Temperature in Kelvin. Default is 298 K.
    xmax : float, optional
        System length in Angstrom. Default: 1000 A. Not used if x_mesh is given.
    x_mesh : np.ndarray, optional
        Custom mesh for the system.
    """

    def __init__(
        self,
        electrolyte: LatticeElectrolyte,
        boundary: Boundary,
        temperature: float = 298,
        xmax: float = 1000,
        x_mesh: Optional[np.ndarray] = None,
    ) -> None:
        self.el = electrolyte
        self.boundary = boundary
        kbt = constants.Boltzmann * temperature
        self.kappa = (
            constants.elementary_charge**2
            / constants.epsilon_0
            / constants.angstrom
            / kbt
        )
        self.kbt_ev = kbt / constants.elementary_charge
        if x_mesh is None:
            self.x_mesh = get_default_mesh(self.boundary, xmax)
        else:
            self.x_mesh = x_mesh

    def ode_rhs(
        self, x: np.ndarray, y: np.ndarray  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """
        Right-hand side of the ODE system for the Poisson-Boltzmann BVP,

                d(ε·φ')/dx = -κ·Σq_i·n_i

        with ε = ε(φ, φ'). Applying the chain rule gives

            dφ'/dx = -κ · (ion_charge_density + phi_eps_coupling) / eps_eff

        where eps_eff = d(ε·φ')/dφ' = ε + φ'·∂ε/∂φ' is the effective permittivity
        that appears when ε varies with position.

        Parameters
        ----------
        x : np.ndarray
            Spatial coordinate (unused; the ODE has no explicit x dependence).
        y : np.ndarray
            Shape (2, N): y[0] is the dimensionless potential φ = eφ/kT,
            y[1] is its spatial derivative.
        """
        n_ion, n_sol = self.el.number_densities(y[0], y[1])
        ion_vf = self.el.ion_sizes[:, newaxis] * n_ion / self.el.n_site
        sol_vf = self.el.sol_sizes[:, newaxis] * n_sol / self.el.n_site

        # Free ion charge density Σ q_i · n_i
        ion_charge_density = np.sum(self.el.ion_q[:, newaxis] * n_ion, axis=0)

        # term due to dependence of ε on φ via the ion Boltzmann factors inside
        # Z (in n_s).
        phi_eps_coupling = np.sum(
            self.el.sol_dipole[:, newaxis]
            * y[1]
            * n_sol
            * L.langevin_x(self.el.sol_dipole[:, newaxis] * y[1])
            * np.sum(self.el.ion_q[:, newaxis] * ion_vf, axis=0),
            axis=0,
        )

        # Direct Langevin response: ∂(polarization density)/∂E via L'(E)
        deps_direct = np.sum(
            self.el.sol_dipole[:, newaxis] ** 2
            * n_sol
            * L.d_langevin_x(self.el.sol_dipole[:, newaxis] * y[1]),
            axis=0,
        )
        # Lattice response: ∂(polarization density)/∂E via n_s(E) through Z
        deps_lattice = np.sum(
            self.el.sol_dipole[:, newaxis] ** 2
            * n_sol
            * L.langevin_x(self.el.sol_dipole[:, newaxis] * y[1]) ** 2
            * (1 - np.sum(sol_vf, axis=0)),
            axis=0,
        )
        # d(ε·y1)/dy1, this is not the permittivity ε itself
        eps_eff = self.el.min_eps + self.kappa * (deps_direct + deps_lattice)

        dy1 = -self.kappa * (ion_charge_density + phi_eps_coupling) / eps_eff
        return np.vstack([y[1], dy1])

    def boundary_condition(
        self, ya: np.ndarray, yb: np.ndarray, y0: float
    ) -> np.ndarray:
        """
        Apply the Stern layer boundary condition.

        Parameters
        ----------
        ya : np.ndarray
            Solution vector at the left boundary (outer Helmholtz plane).
        yb : np.ndarray
            Solution vector at the right boundary (electrolyte bulk).
        y0 : float
            Dimensionless potential at the electrode.
        """
        return self.boundary.residual(
            ya,
            yb,
            y0,
            eps_diffuse=self.permittivity(ya.reshape(2, 1)).squeeze(),
        )

    def permittivity(self, y: np.ndarray) -> np.ndarray:
        """
        Local relative permittivity ε = min_eps + κ · Σ p_s² · n_s · L(p_s·E)/(p_s·E).

        Parameters
        ----------
        y : np.ndarray
            Shape (2, N): y[0] dimensionless potential, y[1] its derivative.
        """
        return self.el.min_eps + self.kappa * self.el.polarization_density(y[0], y[1])

    def voltammetry(
        self,
        potential: np.ndarray,
        tol: float = 1e-3,
    ) -> VoltammetryResult:
        """
        Perform a numerical solution to a potential sweep for the defined double-layer model.

        Parameters
        ----------
        potential : np.ndarray
            Applied potential array in V.
        tol : float, optional
            Solver tolerance. Default is 1e-3.

        Returns
        -------
        VoltammetryResult
            Results container with potential, surface_charge (µC/cm²),
            capacitance (µF/cm²), and electric_field (V/Å).
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

        electric_field = -y[1, :, 0] * self.kbt_ev
        permittivity = self.permittivity(y[:, :, 0])

        surface_charge = (
            constants.epsilon_0
            * permittivity
            * electric_field
            / constants.angstrom
            * 100
        )
        capacitance = np.gradient(surface_charge, edge_order=2) / np.gradient(potential)

        return VoltammetryResult(
            potential=potential,  # V
            surface_charge=surface_charge,  # µC/cm²
            capacitance=capacitance,  # µF/cm²
            electric_field=electric_field,  # V/Å
        )

    def single_point(
        self,
        potential: float,
        tol: float = 1e-3,
    ) -> SinglePointResult:
        """
        Compute spatial profiles through the double layer at a given potential.

        Parameters
        ----------
        potential : float
            Applied potential at the electrode in V.
        tol : float, optional
            Solver tolerance. Default is 1e-3.

        Returns
        -------
        SinglePointResult
            Profiles of potential, electric field, permittivity, and species
            concentrations (mol/L) as a function of position (Å).
        """

        def _species_concentrations(y, x_l, x_r) -> Dict:
            n_ion, n_sol = self.el.number_densities(y[0], y[1])
            nan_l = np.full(x_l.shape, np.nan)
            nan_r = np.full(x_r.shape, np.nan)
            # Convert Å⁻³ → mol/L
            to_molar = 1.0 / (1e3 * constants.Avogadro * constants.angstrom**3)
            result = {}
            for i, name in enumerate(self.el.ion_names):
                result[name] = np.concatenate([nan_l, n_ion[i] * to_molar, nan_r])
            for i, name in enumerate(self.el.sol_names):
                result[name] = np.concatenate([nan_l, n_sol[i] * to_molar, nan_r])
            return result

        if np.abs(potential) < 0.01:
            sweep_par = np.zeros(1)
        else:
            step = potential / abs(potential) * 0.01 / self.kbt_ev
            sweep_par = np.arange(
                start=0,
                stop=potential / self.kbt_ev + step,
                step=step,
            )

        y = sweep_solve_bvp(
            fun=self.ode_rhs,
            bc=self.boundary_condition,
            x0=self.x_mesh,
            y0=np.zeros((2, len(self.x_mesh))),
            sweep_par=sweep_par,
            sweep_par_start=0,
            tol=tol,
        )[:, -1, :]

        eps_diffuse = self.permittivity(y)
        x_l, y_l, eps_l = self.boundary.left_profile(
            y[:, 0], potential / self.kbt_ev, eps_diffuse=eps_diffuse[0]
        )
        x_r, y_r, eps_r = self.boundary.right_profile(
            y[:, -1], potential / self.kbt_ev, eps_diffuse=eps_diffuse[-1]
        )
        y_all = np.concatenate([y_l, y, y_r[:, ::-1]], axis=1)

        return SinglePointResult(
            x=np.concatenate(
                [
                    x_l,
                    self.x_mesh + x_l[-1],
                    self.x_mesh[-1] + x_l[-1] + x_r[-1] - x_r[::-1],
                ]
            ),
            potential=y_all[0] * self.kbt_ev,
            electric_field=y_all[1] * self.kbt_ev,
            permittivity=np.concatenate([eps_l, eps_diffuse, eps_r[::-1]]),
            concentrations=_species_concentrations(y, x_l, x_r),
        )
