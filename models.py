"""
Implementation of double-layer models
"""
from abc import abstractmethod

import numpy as np
from scipy.integrate import solve_bvp

import constants as C
import spatial_profiles as prf


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


class DoubleLayerModel:
    """
    Base class for an ODE. Makes sure that each class
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

    def dirichlet(self, phi0):
        """
        Return a function to pass to an ODE solver
        """
        return lambda ya, yb: np.array([ya[0] - C.BETA * C.Z * C.E_0 * phi0, yb[0]])

    def odesolve_dirichlet(self,
            x_axis: np.ndarray,
            y_initial: np.ndarray,
            phi0: float,
            tol: float=1e-3):
        """
        Solve the ODE on the specified geometry with specified boundary conditions
        """
        sol = solve_bvp(
            self.ode_rhs,
            self.dirichlet(phi0),
            x_axis,
            y_initial,
            tol=tol,
            max_nodes=int(1e8),
            verbose=1)
        return sol

    @abstractmethod
    def compute_profiles(self, sol) -> prf.SpatialProfilesSolution:
        """
        Convert a dimensionless scipy solution into dimensional spatial
        profiles
        """


class GouyChapman(DoubleLayerModel):
    """
    Gouy-Chapman model, treating ions as point particles obeying Boltzmann statistics.
    See for example Schmickler & Santos' Interfacial Electrochemistry.
    """
    def __init__(self, ion_concentration_molar: float):
        self.c_0 = ion_concentration_molar
        self.n_0 = self.c_0 * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.Z*C.E_0)**2 /
                                   (C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))
        self.name = 'Gouy-Chapman'

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        dy2 = np.sinh(y[0, :])
        return np.vstack([dy1, dy2])

    def get_dimensionless_x_axis(self, x_axis_nm):
        """
        Return dimensionless x-axis using model length scale.
        """
        return self.kappa_debye * 1e-9 * x_axis_nm

    def compute_profiles(self, sol):
        ret = prf.SpatialProfilesSolution(
            x=sol.x / self.kappa_debye * 1e9,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_dict={
                'Cations': self.c_0 * np.exp(-sol.y[0, :]),
                'Anions': self.c_0 * np.exp(sol.y[0, :]),
                'Solvent': np.zeros(sol.x.shape)},
            eps=np.ones(sol.x.shape) * C.EPS_R_WATER,
            name=self.name)
        return ret


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

    def get_dimensionless_x_axis(self, x_axis_nm):
        """
        Return dimensionless x-axis using model length scale.
        """
        return self.kappa_debye * 1e-9 * x_axis_nm

    def compute_profiles(self, sol):
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
            eps=np.ones(sol.x.shape) * C.EPS_R_WATER,
            name=self.name)
        return ret

class Abrashkin(DoubleLayerModel):
    """
    Abrashkin / Gongadze & Iglic / Huang model
    """
    def __init__(self, ion_concentration_molar: float, alpha_c: float, alpha_a: float, eps_r_opt=C.N_WATER**2):
        self.c_0 = ion_concentration_molar
        self.n_0 = self.c_0 * 1e3 * C.N_A

        self.alpha_c = alpha_c
        self.alpha_a = alpha_a
        self.n_max = C.C_WATER_BULK * 1e3 * C.N_A
        self.n_s_0 = self.n_max - alpha_c * self.n_0 - alpha_a * self.n_0
        self.kappa_debye = np.sqrt(2*C.BETA*self.n_max*(C.Z*C.E_0)**2 /
                                   (C.EPS_R_WATER*C.EPS_0))

        self.chi = self.n_0 / self.n_max
        self.chi_s = self.n_s_0 / self.n_max

        self.eps_r_opt = eps_r_opt

        p_water = np.sqrt(3 * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / (C.BETA * self.n_s_0))
        self.p_tilde = p_water * self.kappa_debye / (C.Z * C.E_0)

        self.name = f'DPB {self.c_0:.3f}M {alpha_c:.1f}-{alpha_a:.1f}'

    def densities(self, sol_y):
        """
        Compute cation, anion and solvent densities.
        """
        bf_c = np.exp(-sol_y[0, :])
        bf_a = np.exp(+sol_y[0, :])
        bf_s = sinh_x_over_x(self.p_tilde * sol_y[1, :])
        denom = self.chi_s * bf_s + self.alpha_c * self.chi * bf_c + self.alpha_a * self.chi * bf_a
        n_cat = self.n_0 * bf_c / denom
        n_an  = self.n_0 * bf_a / denom
        n_sol = self.n_s_0 * bf_s / denom
        return n_cat, n_an, n_sol

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        n_cat, n_an, n_sol = self.densities(y)

        numer1 = n_an - n_cat
        numer2 = self.p_tilde * y[1, :] * langevin_x(self.p_tilde * y[1, :]) * \
            (self.alpha_a * n_an - self.alpha_c * n_cat) * n_sol/self.n_max
        eps_ratio = self.kappa_debye ** 2 * C.EPS_0 / (C.Z * C.E_0)**2 / C.BETA
        denom1 = self.p_tilde**2 * n_sol * d_langevin_x(self.p_tilde * y[1, :])
        denom2 = self.p_tilde**2 * langevin_x(self.p_tilde * y[1, :])**2 * (self.alpha_c * n_cat + self.alpha_a * n_an) * n_sol/self.n_max

        dy2 = (numer1 + numer2) / (eps_ratio + denom1 + denom2)
        return np.vstack([dy1, dy2])

    def get_dimensionless_x_axis(self, x_axis_nm):
        """
        Return dimensionless x-axis using model length scale.
        """
        return self.kappa_debye * 1e-9 * x_axis_nm

    def permittivity(self, sol_y):
        """
        Compute the permittivity using the electric field
        n_sol: solvent number density
        y_1: dimensionless electric field
        """
        sol_y = np.atleast_1d(sol_y).reshape(2, -1)
        _, _, n_sol = self.densities(sol_y)
        return self.eps_r_opt + \
               1/2 * C.EPS_R_WATER * self.p_tilde**2 * n_sol / self.n_max * \
               langevin_x_over_x(self.p_tilde * sol_y[1, :])

    def compute_profiles(self, sol):
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
            eps=self.permittivity(sol.y),
            name=self.name)

        return ret