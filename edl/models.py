"""
Implementation of double-layer models
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d

from edl import constants as C
from edl.constants import N_WATER

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
        c_cat_molar: float,
        gamma_c: float,
        gamma_a: float,
        gamma_h: float,
        gamma_oh: float,
        eps_r_opt=C.N_WATER**2,
    ) -> None:
        # Nondimensionalization
        self.n_max = C.C_WATER_BULK * 1e3 * C.N_A
        self.kappa = np.sqrt(
            C.BETA * self.n_max * (C.E_0) ** 2 / (C.EPS_0 * C.EPS_R_WATER)
        )

        # Species
        self.c_cat_bulk_molar = c_cat_molar
        self.gammas = np.array([gamma_h, gamma_oh, gamma_c, gamma_a, 1]).reshape(5, 1)
        self.charge = np.array([+1, -1, +1, -1, 0]).reshape(5, 1)

        # Dielectrics
        self.eps_r_opt = eps_r_opt
        p_water = np.sqrt(
            3 * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / (C.BETA * self.n_max)
        )
        self.p_tilde = p_water * self.kappa / (C.Z * C.E_0)

    def bulk_densities(self, p_h: float):
        """
        Compute bulk cation, anion and solvent densities.
        """
        # Compute bulk number densities
        c_bulk = np.zeros((5, 1))
        c_bulk[0] = 10 ** (-p_h)  # [H+]
        c_bulk[1] = 10 ** (-C.PKW + p_h)  # [OH-]
        c_bulk[2] = self.c_cat_bulk_molar  # [Cat]
        c_bulk[3] = c_bulk[0] + c_bulk[2] - c_bulk[1]  # [An]
        c_bulk[4] = C.C_WATER_BULK - np.sum(self.gammas * c_bulk)  # [H2O]
        n_bulk = c_bulk * 1e3 * C.N_A
        return n_bulk

    def boltzmann_factors(self, sol_y):
        """
        Compute cation, anion and solvent Boltzmann factors.
        """
        bf_pos = np.exp(-sol_y[0, :])
        bf_neg = np.exp(+sol_y[0, :])
        bf_sol = L.sinh_x_over_x(self.p_tilde * sol_y[1, :])
        bfs = np.array([bf_pos, bf_neg, bf_pos, bf_neg, bf_sol])  # shape (5, ...)
        return bfs

    def densities(self, sol_y: np.ndarray, p_h: float):
        """
        Compute cation, anion and solvent densities.
        """

        def _get_denominator():
            """
            Get denominator of densities
            """
            n_bulk = self.bulk_densities(p_h)
            chi = n_bulk / self.n_max

            bfs = self.boltzmann_factors(sol_y)
            denom = np.sum(self.gammas * chi * bfs, axis=0)

            return denom

        n_profile = (
            self.bulk_densities(p_h)
            * self.boltzmann_factors(sol_y)
            / _get_denominator()
        )

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

    def ode_rhs(self, x, y, p_h):  # pylint: disable=unused-argument
        """
        Function to pass to ODE solver, specifying the right-hand side of the
        system of dimensionless 1st order ODE's that we solve.
        x: dimensionless x-axis of length n, i.e. kappa (1/m) times x-position (m).
        y: dimensionless potential phi and dphi/dx, size 2 x n.
        """
        dy1 = y[1, :]
        n_arr = self.densities(y, p_h)

        numer1 = np.sum(-self.charge * n_arr, axis=0)
        numer2 = (
            self.p_tilde
            * y[1, :]
            * L.langevin_x(self.p_tilde * y[1, :])
            * np.sum(n_arr * -self.charge * self.gammas, axis=0)
            * n_arr[4]
            / self.n_max
        )
        denom1 = self.n_max * self.eps_r_opt / C.EPS_R_WATER
        denom2 = self.p_tilde**2 * n_arr[4] * L.d_langevin_x(self.p_tilde * y[1, :])
        denom3 = (
            self.p_tilde**2
            * L.langevin_x(self.p_tilde * y[1, :]) ** 2
            * np.sum(n_arr * self.charge**2 * self.gammas, axis=0)
            * n_arr[4]
            / self.n_max
        )

        dy2 = (numer1 + numer2) / (denom1 + denom2 + denom3)
        return np.vstack([dy1, dy2])

    def ode_lambda(self, p_h):
        """
        Get an ODE RHS function to pass to Scipy's solve_bvp; using lambda gives
        the possibility to include specific parameters like pH in derived classes
        """
        return lambda x, y: self.ode_rhs(x, y, p_h)

    def get_stern_layer_thickness(self, phi0):
        """
        Calculate the thickness of the Stern layer as half of the effective ion size
        """
        half_ctrion_thickness = None
        if phi0 < 0:
            half_ctrion_thickness = 1 / 2 * (self.gammas[2, 0] / self.n_max) ** (1 / 3)
        else:
            half_ctrion_thickness = 1 / 2 * (self.gammas[3, 0] / self.n_max) ** (1 / 3)
        # if self.gammas[2, 0] == 7:
        #     half_ctrion_thickness = 5.8e-10
        # elif self.gammas[2, 0] == 6:
        #     half_ctrion_thickness = 5.2e-10
        # elif self.gammas[2, 0] == 5:
        #     half_ctrion_thickness = 4.1e-10
        # else:
        #     half_ctrion_thickness = 3.5e-10
        return half_ctrion_thickness

    def potential_sequential_solve(
        self,
        ode_lambda: callable,
        potential: np.ndarray,
        tol: float = 1e-3,
        p_h: float = 7,
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
                ode_lambda(p_h),
                self.dirichlet_lambda(phi),
                x_axis,
                y_initial,
                tol=tol,
                max_nodes=int(1e8),
                verbose=0,
            )
            profile_df = self.compute_profiles(sol, p_h)
            max_res[i] = np.max(sol.rms_residuals)
            df_list.append(profile_df)

            x_axis = sol.x
            y_initial = sol.y

        print(
            f"Sweep from {potential[0]:.2f}V to {potential[-1]:.2f}V. "
            + f"Maximum relative residual: {np.max(max_res):.5e}."
        )
        return df_list

    def spatial_profiles(
        self, phi0: float, p_h: float = 7, tol: float = 1e-3
    ) -> pd.DataFrame:
        """
        Get spatial profiles solution struct.
        """
        sign = phi0 / abs(phi0)
        profiles = self.potential_sequential_solve(
            self.ode_lambda,
            np.arange(0, phi0 + sign * 0.01, sign * 0.01),
            tol=tol,
            p_h=p_h,
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

    def entropy(self, sol_y, p_h: float):
        """
        Calculate the volumetric entropy density s/kb
        """
        n_arr = self.densities(sol_y, p_h)

        bpe = self.p_tilde * sol_y[1, :]
        select = np.abs(bpe) > 1e-6

        s_over_nkb = np.zeros(sol_y[1, :].shape)
        s_over_nkb[select] = (
            1
            - bpe[select] / np.tanh(bpe[select])
            + np.log(np.sinh(bpe[select]) / bpe[select])
        )

        return s_over_nkb * n_arr[4]

    def grad_pressure(self, x_nm, rho, efield, eps):
        """
        Calculate dP/dx according to Landstorfer & Muller 2022
        """
        return rho * efield + (eps - 1) * C.EPS_0 * efield * np.gradient(
            efield, x_nm * 1e-9
        )

    def compute_profiles(self, sol, p_h: float):
        """
        Compute spatial profiles
        """
        n_arr = self.densities(sol.y, p_h)
        d_stern_m = self.get_stern_layer_thickness(sol.y[0, 0])

        diffuse = pd.DataFrame(
            {
                "x": sol.x / self.kappa * 1e9 + d_stern_m * 1e9,
                "phi": sol.y[0, :] / (C.BETA * C.Z * C.E_0),
                "efield": -sol.y[1, :] * self.kappa / (C.BETA * C.Z * C.E_0),
                "protons": n_arr[0] / 1e3 / C.N_A,
                "hydroxide": n_arr[1] / 1e3 / C.N_A,
                "cations": n_arr[2] / 1e3 / C.N_A,
                "anions": (n_arr[1] + n_arr[3]) / 1e3 / C.N_A,
                "solvent": n_arr[4] / 1e3 / C.N_A,
                "eps": self.permittivity(sol.y, n_arr[4]),
                "charge density": C.E_0 * np.sum(n_arr * self.charge, axis=0),
                "entropy": self.entropy(sol.y, p_h),
            }
        )

        grad_pressure = self.grad_pressure(
            diffuse["x"],
            diffuse["charge density"],
            diffuse["efield"],
            diffuse["eps"],
        )
        diffuse["pressure"] = cumulative_trapezoid(
            grad_pressure[::-1], x=diffuse["x"][::-1] * 1e-9, initial=0
        )[::-1]

        x_stern_m = np.linspace(0, d_stern_m, 10)
        stern = pd.DataFrame(
            {
                "x": x_stern_m * 1e9,
                "phi": diffuse["phi"][0]
                + diffuse["efield"][0] * (d_stern_m - x_stern_m),
                "efield": np.ones(x_stern_m.shape) * diffuse["efield"][0],
                "protons": np.zeros(x_stern_m.shape) * np.nan,
                "hydroxide": np.zeros(x_stern_m.shape) * np.nan,
                "cations": np.zeros(x_stern_m.shape) * np.nan,
                "anions": np.zeros(x_stern_m.shape) * np.nan,
                "solvent": C.C_WATER_BULK,
                "eps": diffuse["eps"][0],
                "charge density": np.zeros(x_stern_m.shape),
                "pressure": np.ones(x_stern_m.shape) * diffuse["pressure"][0],
                "entropy": np.ones(x_stern_m.shape) * diffuse["entropy"][0],
            }
        )

        complete = pd.concat([stern, diffuse])

        complete = complete.reset_index()

        return complete

    def insulator_bc_lambda(self, p_h):
        """
        Return a boundary condition function to pass to scipy's solve_bvp
        """
        return lambda ya, yb: self.insulator_bc(ya, yb, p_h)

    def insulator_bc(self, ya, yb, p_h):
        """
        Boundary condition
        """
        # pylint: disable=invalid-name
        n_arr = self.densities(ya.reshape(2, 1), p_h)
        eps_r = self.permittivity(ya.reshape(2, 1), np.atleast_1d(n_arr[4]))

        n_bulk_arr = self.bulk_densities(p_h)
        c_bulk_arr = n_bulk_arr / 1e3 / C.N_A

        KB = 10 ** (-14) / C.K_SILICA_A

        left = eps_r * ya[
            1
        ] - self.kappa * C.EPS_R_WATER * C.N_SITES_SILICA / self.n_max * c_bulk_arr[
            1
        ] / (
            c_bulk_arr[1]
            + KB
            * np.exp(
                -ya[0] + ya[1] * self.kappa * self.get_stern_layer_thickness(ya[0])
            )
        )

        right = yb[0]

        return np.array([left.squeeze(), right])

    def insulator_sequential_solve(
        self, ph_range: np.ndarray, tol: float = 1e-3
    ) -> list[pd.DataFrame]:
        """
        Sweep over a boundary condition parameter array (potential, pH) and use the previous
        solution as initial condition for the next. The initial condition assumes that we
        start at the PZC.

        Returns: charge for each parameter; list of SpatialProfilesSolutions
        """
        profile_list = []
        max_res = np.zeros(ph_range.shape)

        x_axis = T.create_mesh_m(100, 1000) * self.kappa
        y_initial = np.zeros((2, x_axis.shape[0]))

        for i, p_h in enumerate(ph_range):
            sol = solve_bvp(
                self.ode_lambda(p_h),
                self.insulator_bc_lambda(p_h),
                x_axis,
                y_initial,
                tol=tol,
                max_nodes=int(1e8),
                verbose=0,
            )
            prf = self.compute_profiles(sol, p_h)
            max_res[i] = np.max(sol.rms_residuals)
            profile_list.append(prf)

            x_axis = sol.x
            y_initial = sol.y

        print(
            f"Sweep from pH {ph_range[0]:.2f} to {ph_range[-1]:.2f}. "
            + f"Maximum relative residual: {np.max(max_res):.5e}."
        )
        return profile_list

    def ph_sweep(self, ph_range: np.ndarray, tol: float = 1e-3) -> pd.DataFrame:
        """
        Numerical solution to a potential sweep for a defined double-layer model.
        """
        # Find pH closest to PZC
        ph_pzc = -1 / 2 * np.log10(C.K_SILICA_A * C.K_SILICA_B)
        i_pzc = np.argmin(np.abs(ph_range - ph_pzc)).squeeze()

        profiles_neg = self.insulator_sequential_solve(ph_range[i_pzc::-1], tol)
        profiles_pos = self.insulator_sequential_solve(ph_range[i_pzc::1], tol)

        all_profiles = profiles_neg[::-1] + profiles_pos[1:]

        # Create dataframe with interface values (iloc[0]) to return
        sweep_df = pd.concat(
            [pd.DataFrame(profile.iloc[0]).transpose() for profile in all_profiles],
            ignore_index=True,
        )

        sweep_df["ph"] = ph_range
        sweep_df["charge"] = sweep_df["efield"] * C.EPS_0 * sweep_df["eps"]
        if "phi" in sweep_df.columns:
            sweep_df.rename({"phi": "phi2"}, inplace=True, axis=1)
            sweep_df["phi0"] = sweep_df["phi2"] + sweep_df[
                "efield"
            ] * self.get_stern_layer_thickness(-1)
        if "x" in sweep_df.columns:
            sweep_df.drop("x", inplace=True, axis=1)

        return sweep_df

    def insulator_spatial_profiles(self, p_h: float, tol: float = 1e-3) -> pd.DataFrame:
        """
        Get spatial profiles solution struct.
        """
        ph_pzc = -1 / 2 * np.log10(C.K_SILICA_A * C.K_SILICA_B)
        sign = (p_h - ph_pzc) / abs(p_h - ph_pzc)
        profiles = self.insulator_sequential_solve(
            np.arange(ph_pzc, p_h, sign * 0.1), tol
        )
        return profiles[-1]

    def potential_sweep(
        self, potential: np.ndarray, tol: float = 1e-3, p_h: float = 7
    ) -> pd.DataFrame:
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

        profiles_neg = self.potential_sequential_solve(
            self.ode_lambda, potential[i_pzc::-1], tol=tol, p_h=p_h
        )
        profiles_pos = self.potential_sequential_solve(
            self.ode_lambda, potential[i_pzc::1], tol
        )

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
        sweep_df["sol_2"] = np.array(
            [_get_stern_qty(profile, "solvent") for profile in all_profiles]
        )

        return sweep_df


class ExplicitStern(DoubleLayerModel):
    def __init__(
        self,
        c_cat_molar: float,
        gamma_c: float,
        gamma_a: float,
        x_2: float,
        eps_r_opt=C.N_WATER**2,
    ) -> None:
        super().__init__(c_cat_molar, gamma_c, gamma_a, 0, 0, eps_r_opt)
        self.x_2 = x_2
        # self.p_tilde = 0

    # def boltzmann_factors(self, sol_y):
    #     """
    #     Compute cation, anion and solvent Boltzmann factors.
    #     """
    #     bf_pos = np.exp(-sol_y[0, :])
    #     bf_neg = np.exp(+sol_y[0, :])
    #     bf_sol = np.ones(sol_y[1, :].shape)
    #     bfs = np.array([bf_pos, bf_neg, bf_pos, bf_neg, bf_sol])  # shape (5, ...)
    #     return bfs

    # def densities(self, sol_y: np.ndarray, p_h: float):
    #     """
    #     Compute cation, anion and solvent densities.
    #     """
    #     n_profile = self.bulk_densities(p_h) * self.boltzmann_factors(sol_y)

    #     return n_profile

    def get_stern_layer_thickness(self, phi0):
        return self.x_2


class PoissonBoltzmann(DoubleLayerModel):
    """
    Gouy-Chapman-Stern model
    """

    def __init__(
        self,
        c_cat_molar: float,
        gamma_c: float,
        gamma_a: float,
        gamma_h: float,
        gamma_oh: float,
    ) -> None:
        super().__init__(
            c_cat_molar, gamma_c, gamma_a, gamma_h, gamma_oh, C.EPS_R_WATER
        )

        self.p_tilde = 0

    def boltzmann_factors(self, sol_y):
        """
        Compute cation, anion and solvent Boltzmann factors.
        """
        bf_pos = np.exp(-sol_y[0, :])
        bf_neg = np.exp(+sol_y[0, :])
        bf_sol = np.ones(sol_y[1, :].shape)
        bfs = np.array([bf_pos, bf_neg, bf_pos, bf_neg, bf_sol])  # shape (5, ...)
        return bfs

    def densities(self, sol_y: np.ndarray, p_h: float):
        """
        Compute cation, anion and solvent densities.
        """
        n_profile = self.bulk_densities(p_h) * self.boltzmann_factors(sol_y)

        return n_profile


class LangevinPoissonBoltzmann(DoubleLayerModel):
    """
    Gouy-Chapman-Stern model
    """

    def __init__(
        self,
        c_cat_molar: float,
        gamma_c: float,
        gamma_a: float,
        gamma_h: float,
        gamma_oh: float,
        eps_r_opt=C.N_WATER**2,
    ) -> None:
        super().__init__(c_cat_molar, gamma_c, gamma_a, gamma_h, gamma_oh, eps_r_opt)

    def boltzmann_factors(self, sol_y):
        """
        Compute cation, anion and solvent Boltzmann factors.
        """
        bf_pos = np.exp(-sol_y[0, :])
        bf_neg = np.exp(+sol_y[0, :])
        bf_sol = np.ones(sol_y[1, :].shape)
        bfs = np.array([bf_pos, bf_neg, bf_pos, bf_neg, bf_sol])  # shape (5, ...)
        return bfs

    def densities(self, sol_y: np.ndarray, p_h: float):
        """
        Compute cation, anion and solvent densities.
        """
        n_profile = self.bulk_densities(p_h) * self.boltzmann_factors(sol_y)

        return n_profile


class Bikerman(DoubleLayerModel):
    """
    Gouy-Chapman-Stern model
    """

    def __init__(
        self,
        c_cat_molar: float,
        gamma_c: float,
        gamma_a: float,
        gamma_h: float,
        gamma_oh: float,
    ) -> None:
        super().__init__(
            c_cat_molar, gamma_c, gamma_a, gamma_h, gamma_oh, C.EPS_R_WATER
        )
        self.p_tilde = 0

    def boltzmann_factors(self, sol_y):
        """
        Compute cation, anion and solvent Boltzmann factors.
        """
        bf_pos = np.exp(-sol_y[0, :])
        bf_neg = np.exp(+sol_y[0, :])
        bf_sol = np.ones(sol_y[1, :].shape)
        bfs = np.array([bf_pos, bf_neg, bf_pos, bf_neg, bf_sol])  # shape (5, ...)
        return bfs

    def densities(self, sol_y: np.ndarray, p_h: float):
        """
        Compute cation, anion and solvent densities.
        """

        def _get_denominator():
            """
            Get denominator of densities
            """
            n_bulk = self.bulk_densities(p_h)
            chi = n_bulk / self.n_max

            bfs = self.boltzmann_factors(sol_y)
            denom = np.sum(self.gammas * chi * bfs, axis=0)

            return denom

        n_profile = (
            self.bulk_densities(p_h)
            * self.boltzmann_factors(sol_y)
            / _get_denominator()
        )

        return n_profile
