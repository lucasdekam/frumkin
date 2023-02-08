"""
Implementation of double-layer models
"""
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_bvp

import constants as C
import spatial_profiles as prf


def get_odesol(x_axis, ode_rhs, boundary_condition, y_initial, model_name, spec_string, verbose=True, force_recalculation=True):
    """
    Solve the ODE system or load the solution if there is one already
    """
    # Make directory for solutions if there is none existing
    parent_folder_path = './solutions/'
    folder_path = os.path.join(parent_folder_path, model_name)
    if not model_name in os.listdir(parent_folder_path):
        os.mkdir(folder_path)
    pickle_name = f'sol_{model_name}_{spec_string}.pkl'
    pickle_path = os.path.join(folder_path, pickle_name)

    # Solve or load solution
    sol = None
    if pickle_name not in os.listdir(folder_path) or force_recalculation:
        sol = solve_bvp(
            ode_rhs,
            boundary_condition,
            x_axis,
            y_initial,
            # tol=1e-5,
            max_nodes=int(1e8),
            verbose=verbose)
        with open(pickle_path, 'wb') as file:
            pickle.dump(sol, file)
        if verbose > 0:
            print(f'ODE problem solved and saved under {pickle_path}.')
    else:
        if verbose > 0:
            print(f'File {pickle_name} found.')
        with open(pickle_path, 'rb') as file:
            sol = pickle.load(file)

    return sol


def langevin_x_over_x(x): # pylint: disable=invalid-name
    """
    Returns L(x)/x, where L(x) is the Langevin function:
    L(x) = 1/tanh(x) - 1/x
    For small x, the function value is 1/3
    """
    ret = np.zeros(np.atleast_1d(x).shape)
    select = np.abs(x) > 1e-4
    ret[select] = (1/np.tanh(x[select]) - 1/x[select])/x[select]
    ret[~select] = 1/3
    return ret


def langevin_x(x): #pylint: disable=invalid-name
    """
    Returns L(x), where L(x) is the Langevin function:
    L(x) = 1/tanh(x) - 1/x
    For small x, the function value is zero
    """
    ret = np.zeros(np.atleast_1d(x.shape))
    select = np.abs(x) > 1e-9
    ret[select] = 1/np.tanh(x[select]) - 1/x[select]
    return ret


def d_langevin_x(x): #pylint: disable=invalid-name
    """
    Returns the derivative of the Langevin function to x:
    dL/dx = 1/x^2 - 1/sinh^2 x
    For small x, the function value is 1/3
    """
    ret = np.zeros(np.atleast_1d(x.shape))

    select_small = np.abs(x) < 1e-4
    select_big = np.abs(x) > 1e2
    select_normal = ~select_small * ~select_big

    ret[select_small] = 1/3
    ret[select_normal] = 1/x[select_normal]**2 - 1/np.sinh(x[select_normal])**2
    ret[select_big] = 0
    return ret


def sinh_x_over_x(x): #pylint: disable=invalid-name
    """
    Returns sinh(x)/x. For small x, the function value is one
    """
    ret = np.zeros(x.shape)

    select_small = np.abs(x) < 1e-9
    select_big = np.abs(x) > 1e2
    select_normal = ~select_small * ~select_big

    ret[select_small] = 1
    ret[select_normal] = np.sinh(x[select_normal])/(x[select_normal])
    ret[select_big] = 2e41
    return ret


class DoubleLayerModel(ABC):
    """
    Abstract base class for an ODE. Makes sure that each class
    has a function to pass to an ODE solver, and a name property
    """
    @abstractmethod
    def ode_rhs(self, x, y):  # pylint: disable=unused-argument, invalid-name
        """
        Function to pass to ODE solver, specifying the right-hand side of the
        system of dimensionless 1st order ODE's that we solve.
        x: dimensionless x-axis of length n, i.e. kappa (1/m) times x-position (m).
        y: dimensionless potential phi and dphi/dx, size 2 x n.
        """

    @abstractmethod
    def solve_dirichlet(self,
            x_axis_nm: np.ndarray,
            phi0: float,
            force_recalculation=True):
        """
        Solve the ODE on the specified geometry with specified boundary conditions
        """


class Borukhov(DoubleLayerModel):
    """
    Model developed by Borukhov, Andelman and Orland, modifying the Guy-Chapman model to
    take finite ion size into account.
    https://doi.org/10.1016/S0013-4686(00)00576-4
    """
    def __init__(self, ion_concentration_molar: float, a_m: float):
        self.c_0 = ion_concentration_molar
        self.n_0 = self.c_0 * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.Z*C.E_0)**2 /
                                   (C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))
        self.chi_0 = 2 * a_m ** 3 * self.n_0
        self.n_max = 1/a_m**3
        self.name = 'Borukhov'

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        dy2 = np.sinh(y[0, :]) / (1 - self.chi_0 + self.chi_0 * np.cosh(y[0, :]))
        return np.vstack([dy1, dy2])

    def dirichlet(self, phi0):
        return lambda ya, yb: np.array([ya[0] - C.BETA * C.Z * C.E_0 * phi0, yb[0]])

    def solve_dirichlet(self, x_axis_nm: np.ndarray, phi0: float, force_recalculation=True):
        y_initial=np.zeros((2, x_axis_nm.shape[0]))
        y_initial[0, :] = C.BETA * C.Z * C.E_0 * phi0 * np.exp(-x_axis_nm*1e-9*self.kappa_debye)
        y_initial[1, :] = np.gradient(y_initial[0, :], x_axis_nm*1e-9*self.kappa_debye)

        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            self.dirichlet(phi0),
            y_initial,
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_Dirichlet{phi0:.2f}',
            force_recalculation=force_recalculation)

        bf_c = np.exp(-sol.y[0, :])
        bf_a = np.exp(sol.y[0, :])
        denom = 1 - self.chi_0 + self.chi_0 * np.cosh(sol.y[0, :])

        # Return solution struct
        ret = prf.SpatialProfilesSolution(
            x=sol.x / self.kappa_debye * 1e9,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_dict={
                'Cations': self.c_0 * bf_c / denom,
                'Anions': self.c_0 * bf_a / denom,
                'Solvent': np.zeros(sol.x.shape)
            },
            c_sites=self.n_max/1e3/C.N_A,
            eps=np.ones(sol.x.shape) * C.EPS_R_WATER,
            name=self.name)
        return ret, sol.rms_residuals


class GongadzeIglic(DoubleLayerModel):
    """
    Gongadze & Iglic model
    """
    def __init__(self, ion_concentration_molar: float, alpha_cat: float, alpha_an: float, eps_r_opt=C.N_WATER**2):
        self.c_0 = ion_concentration_molar
        self.n_0 = self.c_0 * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.Z*C.E_0)**2 /
                                   (C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))
        self.n_s_0 = C.C_WATER_BULK * 1e3 * C.N_A

        self.alpha_cat = alpha_cat
        self.alpha_an = alpha_an
        self.n_max = self.n_s_0 + alpha_cat * self.n_0 + alpha_an * self.n_0
        print(1/self.n_max**(1/3))

        self.chi = self.n_0 / self.n_max
        self.chi_s = self.n_s_0 / self.n_max

        self.g_1 = 1# (2 + C.N_WATER ** 2) / 3
        self.g_2 = 1# (2 + C.N_WATER ** 2) / 2

        self.eps_r_opt = eps_r_opt

        p_water = np.sqrt(3 * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / (self.g_1 * self.g_2 * C.BETA * self.n_s_0))
        print(p_water)
        self.p_tilde = p_water * self.kappa_debye / (C.Z * C.E_0)

        self.name = f'GongadzeIglic {alpha_cat}-{alpha_an}'

    def densities(self, sol_y):
        """
        Compute cation, anion and solvent densities.
        """
        bf_c = np.exp(np.minimum(-sol_y[0, :], 1e2))
        bf_a = np.exp(np.minimum(+sol_y[0, :], 1e2))
        bf_s = sinh_x_over_x(self.g_2 * self.p_tilde * sol_y[1, :])
        denom = self.chi_s * bf_s + self.alpha_cat * self.chi * bf_c + self.alpha_an * self.chi * bf_a
        n_cat = self.n_0 * bf_c / denom
        n_an  = self.n_0 * bf_a / denom
        n_sol = self.n_s_0 * bf_s / denom
        return n_cat, n_an, n_sol

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        n_cat, n_an, n_sol = self.densities(y)

        numer = 1 + self.g_1 * self.p_tilde * y[1, :] * langevin_x(self.g_2 * self.p_tilde * y[1, :]) * n_sol/self.n_max
        eps_ratio = self.eps_r_opt / C.EPS_R_WATER
        denom1 = self.g_1 * self.g_2 * self.p_tilde**2 * langevin_x(self.g_2 * self.p_tilde * y[1, :])**2 * (n_cat + n_an)/self.n_0 * n_sol/self.n_max
        denom2 = self.g_1 * self.g_2 * self.p_tilde**2 * n_sol/self.n_0 * d_langevin_x(self.g_2 * self.p_tilde * y[1, :])

        mulfac = numer / (2 * eps_ratio + denom1 + denom2)

        dy2 = - (n_cat - n_an)/self.n_0 * mulfac
        return np.vstack([dy1, dy2])

    def permittivity(self, sol_y):
        """
        Compute the permittivity using the electric field
        n_sol: solvent number density
        y_1: dimensionless electric field
        """
        sol_y = np.atleast_1d(sol_y).reshape(2, -1)
        _, _, n_sol = self.densities(sol_y)
        return self.eps_r_opt + \
               1/2 * C.EPS_R_WATER * self.p_tilde**2 * self.g_1 * self.g_2 * n_sol / self.n_0 * \
               langevin_x_over_x(self.g_2 * self.p_tilde * sol_y[1, :])

    def neumann(self, charge_coulm2):
        """
        Neumann BC function
        """
        return lambda ya, yb: np.array([ya[1] + C.BETA * C.Z * C.E_0 / self.kappa_debye * charge_coulm2 / C.EPS_0 / self.permittivity(ya).squeeze(), yb[0]])

    def dirichlet(self, phi0):
        return lambda ya, yb: np.array([ya[0] - C.BETA * C.Z * C.E_0 * phi0, yb[0]])

    def solve_dirichlet(self,
            x_axis_nm: np.ndarray,
            phi0: float,
            force_recalculation=True):
        y_initial=np.zeros((2, x_axis_nm.shape[0]))
        y_initial[0, :] = C.BETA * C.Z * C.E_0 * phi0 * np.exp(-x_axis_nm*1e-9*self.kappa_debye)
        y_initial[1, :] = np.gradient(y_initial[0, :], x_axis_nm*1e-9*self.kappa_debye)

        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            self.dirichlet(phi0),
            y_initial,
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_Dirichlet{phi0:.2f}',
            force_recalculation=force_recalculation)

        n_cat, n_an, n_sol = self.densities(sol.y)

        # Return solution struct
        ret = prf.SpatialProfilesSolution(
            x=sol.x / self.kappa_debye * 1e9,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_dict={
                'Cations': n_cat/1e3/C.N_A,
                'Anions': n_an/1e3/C.N_A,
                'Solvent': n_sol/1e3/C.N_A
            },
            c_sites=np.ones(sol.x.shape) * self.n_max/1e3/C.N_A,
            eps=self.permittivity(sol.y),
            name=self.name)
        return ret, sol.rms_residuals

    def solve_charge(self, x_axis_nm: np.ndarray, charge_coulm2: float, force_recalculation=True):
        """
        Solve with Neumann BC
        """
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            self.neumann(charge_coulm2),
            np.zeros((2, x_axis.shape[0])),
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_Neumann{charge_coulm2}',
            force_recalculation=force_recalculation)

        n_cat, n_an, n_sol = self.densities(sol.y)

        # Return solution struct
        ret = prf.SpatialProfilesSolution(
            x=sol.x / self.kappa_debye * 1e9,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_dict={
                'Cations': n_cat/1e3/C.N_A,
                'Anions': n_an/1e3/C.N_A,
                'Solvent': n_sol/1e3/C.N_A
            },
            c_sites=np.ones(sol.x.shape) * self.n_max/1e3/C.N_A,
            eps=self.permittivity(sol.y),
            name=self.name)
        return ret