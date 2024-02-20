"""
Implementation of double-layer models
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d

from edl import constants as C

from . import constants as C
from . import langevin as L
from . import tools as T


class DoubleLayerModel:
    """
    Base class for an ODE. Implements basic features, but leaves
    RHS of ODE and the definition of number densities abstract.
    """

    def __init__(
        self,
        c_0_molar: float,
        gamma_c: float,
        gamma_a: float,
        eps_r_opt=C.N_WATER**2,
    ) -> None:
        # Nondimensionalization
        self.n_max = C.C_WATER_BULK * 1e3 * C.N_A
        self.kappa = np.sqrt(
            C.BETA * self.n_max * (C.E_0) ** 2 / (C.EPS_0 * C.EPS_R_WATER)
        )

        # Species
        self.c_bulk_molar = c_0_molar
        self.gammas = np.array([gamma_c, gamma_a, 1]).reshape(-1, 1)
        self.charge = np.array([+1, -1, 0]).reshape(-1, 1)

        # Dielectrics
        self.eps_r_opt = eps_r_opt
        p_water = np.sqrt(
            3 * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / (C.BETA * self.n_max)
        )
        self.p_tilde = p_water * self.kappa / (C.Z * C.E_0)

    def densities(self, sol_y: np.ndarray):
        """
        Compute cation, anion and solvent densities.
        """

        def _bulk_densities():
            """
            Compute bulk cation, anion and solvent densities.
            """
            # Compute bulk number densities
            c_bulk = np.zeros((3, 1))
            c_bulk[0] = self.c_bulk_molar  # [Cat]
            c_bulk[1] = self.c_bulk_molar  # [An] + [OH]
            c_bulk[2] = C.C_WATER_BULK - np.sum(self.gammas * c_bulk)  # [H2O]
            n_bulk = c_bulk * 1e3 * C.N_A
            return n_bulk

        def _boltzmann_factors(y):
            """
            Compute cation, anion and solvent Boltzmann factors.
            """
            bf_pos = np.exp(-y[0, :])
            bf_neg = np.exp(+y[0, :])
            bf_sol = L.sinh_x_over_x(self.p_tilde * y[1, :])
            bfs = np.array([bf_pos, bf_neg, bf_sol])  # shape (5, ...)
            return bfs

        def _get_denominator():
            """
            Get denominator of densities
            """
            n_bulk = _bulk_densities()
            chi = n_bulk / self.n_max

            bfs = _boltzmann_factors(sol_y)
            denom = np.sum(self.gammas * chi * bfs, axis=0)

            return denom

        n_profile = _bulk_densities() * _boltzmann_factors(sol_y) / _get_denominator()

        return n_profile

    def dirichlet_lambda(self, phi0):
        """
        Return a boundary condition function to pass to scipy's solve_bvp
        """
        return lambda ya, yb: np.array(
            [
                ya[0]
                - C.BETA * C.Z * C.E_0 * phi0
                - ya[1] * self.kappa * self.get_stern_layer_thickness(phi0),
                yb[0],
            ]
        )

    def ode_rhs(self, x, y):  # pylint: disable=unused-argument
        """
        Function to pass to ODE solver, specifying the right-hand side of the
        system of dimensionless 1st order ODE's that we solve.
        x: dimensionless x-axis of length n, i.e. kappa (1/m) times x-position (m).
        y: dimensionless potential phi and dphi/dx, size 2 x n.
        """
        dy1 = y[1, :]
        n_arr = self.densities(y)

        numer1 = np.sum(-self.charge * n_arr, axis=0)
        numer2 = (
            self.p_tilde
            * y[1, :]
            * L.langevin_x(self.p_tilde * y[1, :])
            * np.sum(n_arr * -self.charge * self.gammas, axis=0)
            * n_arr[2]
            / self.n_max
        )
        denom1 = self.n_max * self.eps_r_opt / C.EPS_R_WATER
        denom2 = self.p_tilde**2 * n_arr[2] * L.d_langevin_x(self.p_tilde * y[1, :])
        denom3 = (
            self.p_tilde**2
            * L.langevin_x(self.p_tilde * y[1, :]) ** 2
            * np.sum(n_arr * self.charge**2 * self.gammas, axis=0)
            * n_arr[2]
            / self.n_max
        )

        dy2 = (numer1 + numer2) / (denom1 + denom2 + denom3)
        return np.vstack([dy1, dy2])

    def get_stern_layer_thickness(self, phi0):
        """
        Calculate the thickness of the Stern layer as half of the effective ion size
        """
        half_ctrion_thickness = None
        if phi0 < 0:
            half_ctrion_thickness = 1 / 2 * (self.gammas[0, 0] / self.n_max) ** (1 / 3)
        else:
            half_ctrion_thickness = 1 / 2 * (self.gammas[1, 0] / self.n_max) ** (1 / 3)
        return half_ctrion_thickness

    def potential_sequential_solve(
        self,
        potential: np.ndarray,
        tol: float = 1e-3,
    ) -> list[pd.DataFrame]:
        """
        Sweep over a Dirichlet boundary condition for the potential and use the previous
        solution as initial condition for the next. The initial condition assumes that we
        start at the PZC.

        Returns: charge for each parameter; list of SpatialProfilesSolutions
        """
        df_list = []
        max_res = np.zeros(potential.shape)

        x_axis = T.create_mesh_m(100, 1000) * self.kappa
        y_initial = np.zeros((2, x_axis.shape[0]))

        for i, phi in enumerate(potential):
            sol = solve_bvp(
                self.ode_rhs,
                self.dirichlet_lambda(phi),
                x_axis,
                y_initial,
                tol=tol,
                max_nodes=int(1e8),
                verbose=0,
            )
            profile_df = self.compute_profiles(sol)
            max_res[i] = np.max(sol.rms_residuals)
            df_list.append(profile_df)

            x_axis = sol.x
            y_initial = sol.y

        print(
            f"Sweep from {potential[0]:.2f}V to {potential[-1]:.2f}V. "
            + f"Maximum relative residual: {np.max(max_res):.5e}."
        )
        return df_list

    def spatial_profiles(self, phi0: float, tol: float = 1e-3) -> pd.DataFrame:
        """
        Get spatial profiles solution struct.
        """
        sign = phi0 / abs(phi0)
        profiles = self.potential_sequential_solve(
            np.arange(0, phi0 + sign * 0.01, sign * 0.01),
            tol=tol,
        )
        return profiles[-1]

    def permittivity(self, sol_y: np.ndarray, n_sol: np.ndarray):
        """
        Compute the permittivity using the electric field
        n_sol: solvent number density
        y_1: dimensionless electric field
        """
        return (
            self.eps_r_opt
            + C.EPS_R_WATER
            * self.p_tilde**2
            * n_sol
            / self.n_max
            * L.langevin_x_over_x(self.p_tilde * sol_y[1, :])
        )

    def compute_profiles(self, sol):
        """
        Compute spatial profiles
        """
        n_arr = self.densities(sol.y)
        d_stern_m = self.get_stern_layer_thickness(sol.y[0, 0])

        diffuse = pd.DataFrame(
            {
                "x": sol.x / self.kappa * 1e9 + d_stern_m * 1e9,
                "phi": sol.y[0, :] / (C.BETA * C.Z * C.E_0),
                "efield": -sol.y[1, :] * self.kappa / (C.BETA * C.Z * C.E_0),
                "cations": n_arr[0] / 1e3 / C.N_A,
                "anions": n_arr[1] / 1e3 / C.N_A,
                "solvent": n_arr[2] / 1e3 / C.N_A,
                "eps": self.permittivity(sol.y, n_arr[2]),
            }
        )

        x_stern_m = np.linspace(0, d_stern_m, 10)
        stern = pd.DataFrame(
            {
                "x": x_stern_m * 1e9,
                "phi": diffuse["phi"][0]
                + diffuse["efield"][0] * (d_stern_m - x_stern_m),
                "efield": np.ones(x_stern_m.shape) * diffuse["efield"][0],
                "cations": np.zeros(x_stern_m.shape) * np.nan,
                "anions": np.zeros(x_stern_m.shape) * np.nan,
                "solvent": C.C_WATER_BULK,
                "eps": diffuse["eps"][0],
            }
        )

        complete = pd.concat([stern, diffuse])

        complete = complete.reset_index()

        return complete

    def potential_sweep(self, potential: np.ndarray, tol: float = 1e-3) -> pd.DataFrame:
        """
        Numerical solution to a potential sweep for a defined double-layer model.
        """

        def _get_rp_qty(sol: pd.DataFrame, qty: str):
            phi_func = interp1d(sol["x"], sol[qty])
            return phi_func(C.X_REACTION_PLANE * 1e9)

        def _get_stern_qty(sol: pd.DataFrame, qty: str):
            phi_func = interp1d(sol["x"], sol[qty])
            return phi_func(self.get_stern_layer_thickness(-1) * 1e9)

        # Find potential closest to PZC
        i_pzc = np.argmin(np.abs(potential)).squeeze()

        profiles_neg = self.potential_sequential_solve(potential[i_pzc::-1], tol)
        profiles_pos = self.potential_sequential_solve(potential[i_pzc::1], tol)

        all_profiles = profiles_neg[::-1] + profiles_pos[1:]

        # Create dataframe with interface values (iloc[0]) to return
        sweep_df = pd.concat(
            [pd.DataFrame(profile.iloc[0]).transpose() for profile in all_profiles],
            ignore_index=True,
        )

        sweep_df["phi0"] = potential
        sweep_df["charge"] = sweep_df["efield"] * C.EPS_0 * sweep_df["eps"]
        sweep_df["capacity"] = np.gradient(
            sweep_df["charge"], edge_order=2
        ) / np.gradient(potential)
        if "phi" in sweep_df.columns:
            sweep_df.rename({"phi": "phi2"}, inplace=True, axis=1)
        if "x" in sweep_df.columns:
            sweep_df.drop("x", inplace=True, axis=1)

        sweep_df["phi_rp"] = np.array(
            [_get_rp_qty(profile, "phi") for profile in all_profiles]
        )
        sweep_df["cat_2"] = np.array(
            [_get_stern_qty(profile, "cations") for profile in all_profiles]
        )
        sweep_df["ani_2"] = np.array(
            [_get_stern_qty(profile, "anions") for profile in all_profiles]
        )
        sweep_df["sol_2"] = np.array(
            [_get_stern_qty(profile, "solvent") for profile in all_profiles]
        )

        return sweep_df


class ExplicitStern(DoubleLayerModel):
    """
    Explicitly specify the Stern layer thickness
    """

    def __init__(
        self,
        c_0_molar: float,
        gamma_c: float,
        gamma_a: float,
        x_2: float,
        eps_r_opt=C.N_WATER**2,
    ) -> None:
        super().__init__(c_0_molar, gamma_c, gamma_a, eps_r_opt)
        self.x_2 = x_2

    def get_stern_layer_thickness(self, phi0):
        """
        Calculate the thickness of the Stern layer as half of the effective ion size
        """
        return self.x_2
