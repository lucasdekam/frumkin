"""
Implementation of double-layer models
"""
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_bvp

import constants as C
import boundary_conditions as bc
import spatial_profiles as prf


def get_odesol(x_axis, odefunc, boundary_condition, model_name, spec_string, verbose=False, force_recalculation=False):
    """
    Solve the ODE system or load the solution if there is one already
    """
    y = np.zeros((2, x_axis.shape[0]))

    # Make directory for solutions if there is none existing
    parent_folder_path = './solutions/'
    folder_path = os.path.join(parent_folder_path, model_name)
    if not model_name in os.listdir(parent_folder_path):
        os.mkdir(folder_path)
    pickle_name = f'sol_{model_name}_{spec_string}.pkl'
    pickle_path = os.path.join(folder_path, pickle_name)

    # Solve or load solution
    sol = None
    if pickle_path not in os.listdir(folder_path) or force_recalculation:
        sol = solve_bvp(
            odefunc,
            boundary_condition,
            x_axis,
            y,
            max_nodes=1000000,
            verbose=verbose)
        with open(pickle_path, 'wb') as file:
            pickle.dump(sol, file)
        if verbose:
            print(f'ODE problem solved and saved under {pickle_path}.')
    else:
        if verbose:
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
    ret = (1/np.tanh(x) - 1/x)/x
    ret[np.abs(x) <= 1e-9] = 1/3
    return ret


def rel_permittivity(eps_r_opt: float, n_0: float, n_sol: np.ndarray, p_tilde: float, y_1: np.ndarray):
    """
    Compute the permittivity using the electric field
    eps_r_opt: relative optical/background permittivity
    n_0: ion number density in the bulk (appears because Debye length is length scale)
    n_sol: solvent number density
    p_tilde: dimensionless dipole moment
    y_1: dimensionless electric field
    """
    return eps_r_opt + 1/2 * C.EPS_R_WATER * p_tilde**2 * n_sol / n_0 * langevin_x_over_x(p_tilde * y_1)


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
    def solve(self, x_axis_nm, boundary_conditions):
        """
        Solve the ODE on the specified geometry with specified boundary conditions
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

    def solve(self, x_axis_nm: np.ndarray, boundary_conditions: bc.BoundaryConditions):
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            boundary_conditions.func,
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_{boundary_conditions.get_name()}')

        # Return solution struct
        ret = prf.SpatialProfilesSolution(
            x=sol.x,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_cat=self.c_0 * np.exp(-sol.y[0, :]),
            c_an=self.c_0 * np.exp(sol.y[0, :]),
            c_sol=np.zeros(sol.x.shape),
            eps=np.ones(sol.x.shape) * C.EPS_R_WATER,
            name=self.name)
        return ret


class BorukhovAndelmanOrland(DoubleLayerModel):
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
        self.name = 'Borukhov'

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        dy2 = np.sinh(y[0, :]) / (1 - self.chi_0 + self.chi_0 * np.cosh(y[0, :]))
        return np.vstack([dy1, dy2])

    def solve(self, x_axis_nm: np.ndarray, boundary_conditions: bc.BoundaryConditions):
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            boundary_conditions.func,
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_{boundary_conditions.get_name()}')

        bf_c = np.exp(-sol.y[0, :])
        bf_a = np.exp(sol.y[0, :])
        denom = 1 - self.chi_0 + self.chi_0 * np.cosh(sol.y[0, :])

        # Return solution struct
        ret = prf.SpatialProfilesSolution(
            x=sol.x,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_cat=self.c_0 * bf_c / denom,
            c_an=self.c_0 * bf_a / denom,
            c_sol=np.zeros(sol.x.shape),
            eps=np.ones(sol.x.shape) * C.EPS_R_WATER,
            name=self.name)
        return ret


class AbrashkinAndelmanOrland(DoubleLayerModel):
    """
    Abrashkin's extension of the Borukhov-Andelman-Orland model where
    dipolar solution molecules are taken into account.
    https://doi.org/10.1103/PhysRevLett.99.077801
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, ion_concentration_molar: float, a_m: float, eps_r_opt: float = 1):
        """
        ion_concentration_molar: ion bulk concentration in molar
        a_m: ion diameter in meters
        eps_r_opt: optical/background relative permittivity, default: 1
        """
        self.c_0 = ion_concentration_molar
        self.n_0 = self.c_0 * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.Z*C.E_0)**2 /
                                   (C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))
        self.chi = a_m ** 3 * self.n_0
        self.eps_r_opt = eps_r_opt

        # Number density of vacancy sites in the bulk:
        self.n_s_0 = 1/a_m ** 3 - 2 * self.n_0
        if self.n_s_0 < 0:
            raise ValueError('Ion number density is larger than lattice site density')

        # Computing the dipole moment p and the dimensionless ptilde
        p_water = np.sqrt(3 * C.K_B * C.T * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / self.n_s_0)
        self.p_tilde = p_water * self.kappa_debye / (C.Z * C.E_0)

        self.name = 'Abrashkin'

    def bfactors(self, sol_y):
        """
        Compute cation, anion and solvent Boltzmann factors, and the denominator
        appearing in the expression of the number density profile.
        """
        bf_c = np.exp(-sol_y[0, :])
        bf_a = np.exp(+sol_y[0, :])

        bf_s = np.zeros(sol_y[1, :].shape)
        select = np.abs(self.p_tilde * sol_y[1, :]) > 1e-9
        bf_s[select] = np.sinh(self.p_tilde * sol_y[1, :][select])/(self.p_tilde * sol_y[1, :][select])
        bf_s[~select] = 1

        denom = (1 - 2* self.chi) * bf_s + self.chi * (bf_c + bf_a)
        return bf_c, bf_a, bf_s, denom

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        bf_c, bf_a, bf_s, denom = self.bfactors(y)
        H = 1/2 * C.EPS_R_WATER/self.eps_r_opt * (1-1/bf_s**2) * (1/self.chi - (bf_c + bf_a)/denom)
        dy2 = - 1/2 * (C.EPS_R_WATER/self.eps_r_opt) * (bf_c - bf_a)/denom * y[1, :]**2 / (y[1, :]**2 + H + 1e-60)
        return np.vstack([dy1, dy2])

    def solve(self, x_axis_nm: np.ndarray, boundary_conditions: bc.BoundaryConditions):
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            boundary_conditions.func,
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_{boundary_conditions.get_name()}')

        bf_c, bf_a, bf_s, denom = self.bfactors(sol.y)
        n_cat = self.n_0 * bf_c / denom
        n_an  = self.n_0 * bf_a / denom
        n_sol = self.n_s_0 * bf_s / denom

        # Return solution struct
        ret = prf.SpatialProfilesSolution(
            x=sol.x,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_cat=n_cat/1e3/C.N_A,
            c_an=n_an/1e3/C.N_A,
            c_sol=n_sol/1e3/C.N_A,
            eps=rel_permittivity(self.eps_r_opt, self.n_0, n_sol, self.p_tilde, sol.y[1, :]),
            name=self.name)
        return ret


class Huang(DoubleLayerModel):
    """
    Huang, Chen and Eikerling's extension of Abrashkin's model
    to take into account different solvent molecule and ion sizes.
    https://doi.org/10.1021/acs.jctc.1c00098
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, ion_concentration_molar: float,
                       d_cat_m: float,
                       d_an_m: float,
                       d_sol_m: float,
                       eps_r_opt: float = 1):
        """
        ion_concentration_molar:    ion bulk concentration in molar
        d_cat_m:                    cation diameter in meters
        d_an_m:                     anion diameter in meters
        d_sol_m:                    solvent molecule diameter in meters
        eps_r_opt:                  optical/background relative permittivity, default: 1
        """
        self.c_0 = ion_concentration_molar
        self.n_0 = self.c_0 * 1e3 * C.N_A
        self.n_s_0 = C.C_WATER_BULK * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.Z*C.E_0)**2 /
                                   (C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))
        self.a_m = 0.5e-10 # roughly Bohr radius

        self.gamma_c = (d_cat_m / self.a_m) ** 3
        self.gamma_a = (d_an_m / self.a_m) ** 3
        self.gamma_s = (d_sol_m / self.a_m) ** 3

        self.chi = self.n_0 * self.a_m ** 3
        self.chi_s = self.n_s_0 * self.a_m ** 3
        self.chi_v = 1 - self.gamma_s*self.chi_s - self.gamma_c*self.chi - self.gamma_a*self.chi

        self.eps_r_opt = eps_r_opt

        # Computing the dipole moment p and the dimensionless ptilde
        p_water = np.sqrt(3 * C.K_B * C.T * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / self.n_s_0)
        self.p_tilde = p_water * self.kappa_debye / (C.Z * C.E_0)

        self.name = 'Huang'

    def bfactors(self, sol_y):
        """
        Compute cation, anion and solvent Boltzmann factors, and the denominator
        appearing in the expression of the number density profile.
        """
        bf_c = np.exp(-sol_y[0, :])
        bf_a = np.exp(+sol_y[0, :])

        bf_s = np.zeros(sol_y[1, :].shape)
        select = np.abs(self.p_tilde * sol_y[1, :]) > 1e-9
        bf_s[select] = np.sinh(self.p_tilde * sol_y[1, :][select])/(self.p_tilde * sol_y[1, :][select])
        bf_s[~select] = 1

        denom = self.chi_v + self.gamma_s*self.chi_s*bf_s + self.gamma_c*self.chi*bf_c + self.gamma_a*self.chi*bf_a
        return bf_c, bf_a, bf_s, denom

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        bf_c, bf_a, bf_s, denom = self.bfactors(y)
        n_sol = self.n_s_0 * bf_s / denom
        H = 1/2 * C.EPS_R_WATER/self.eps_r_opt * (1-1/bf_s**2) * n_sol / self.n_0
        dy2 = - 1/2 * (C.EPS_R_WATER/self.eps_r_opt) * (bf_c - bf_a)/denom * y[1, :]**2 / (y[1, :]**2 + H + 1e-60)
        return np.vstack([dy1, dy2])

    def solve(self, x_axis_nm: np.ndarray, boundary_conditions: bc.BoundaryConditions):
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            boundary_conditions.func,
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_{boundary_conditions.get_name()}')

        bf_c, bf_a, bf_s, denom = self.bfactors(sol.y)
        n_cat = self.n_0 * bf_c / denom
        n_an  = self.n_0 * bf_a / denom
        n_sol = self.n_s_0 * bf_s / denom

        # Return solution struct
        ret = prf.SpatialProfilesSolution(
            x=sol.x,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_cat=n_cat/1e3/C.N_A,
            c_an=n_an/1e3/C.N_A,
            c_sol=n_sol/1e3/C.N_A,
            eps=rel_permittivity(self.eps_r_opt, self.n_0, n_sol, self.p_tilde, sol.y[1, :]),
            name=self.name)
        return ret


class HuangSimple(DoubleLayerModel):
    """
    Huang's simplification in 'Cation Overcrowding Effect on the
    Oxygen Evolution Reaction'
    https://doi.org/10.1021/jacsau.1c00315
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, ion_concentration_molar: float,
                       d_cat_m: float,
                       d_an_m: float,
                       eps_r_opt: float = 1):
        """
        ion_concentration_molar:    ion bulk concentration in molar
        d_cat_m:                    cation diameter in meters
        d_an_m:                     anion diameter in meters
        d_sol_m:                    solvent molecule diameter in meters
        eps_r_opt:                  optical/background relative permittivity, default: 1
        """
        self.c_0 = ion_concentration_molar
        self.n_0 = self.c_0 * 1e3 * C.N_A
        self.n_s_0 = C.C_WATER_BULK * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.Z*C.E_0)**2 /
                                   (C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))

        self.n_max = self.n_s_0 + 2 * self.n_0
        self.a_m = (1/self.n_max)**(1/3)

        self.gamma_c = (d_cat_m / self.a_m) ** 3
        self.gamma_a = (d_an_m / self.a_m) ** 3
        self.chi = self.n_0 * self.a_m ** 3

        self.eps_r_opt = eps_r_opt

        # Computing the dipole moment p and the dimensionless ptilde
        p_water = np.sqrt(3 * C.K_B * C.T * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / self.n_s_0)
        self.p_tilde = p_water * self.kappa_debye / (C.Z * C.E_0)

        self.name = 'HuangSimple'

    def bfactors(self, sol_y):
        """
        Compute cation, anion and solvent Boltzmann factors, and the denominator
        appearing in the expression of the number density profile.
        """
        bf_c = np.exp(-sol_y[0, :])
        bf_a = np.exp(+sol_y[0, :])

        bf_s = np.zeros(sol_y[1, :].shape)
        select = np.abs(self.p_tilde * sol_y[1, :]) > 1e-9
        bf_s[select] = np.sinh(self.p_tilde * sol_y[1, :][select])/(self.p_tilde * sol_y[1, :][select])
        bf_s[~select] = 1

        denom = (1 - 2*self.chi)*bf_s + self.gamma_c*self.chi*bf_c + self.gamma_a*self.chi*bf_a
        return bf_c, bf_a, bf_s, denom

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        bf_c, bf_a, bf_s, denom = self.bfactors(y)
        n_sol = self.n_s_0 * bf_s / denom
        H = 1/2 * C.EPS_R_WATER/self.eps_r_opt * (1-1/bf_s**2) * n_sol / self.n_0
        dy2 = - 1/2 * (C.EPS_R_WATER/self.eps_r_opt) * (bf_c - bf_a)/denom * y[1, :]**2 / (y[1, :]**2 + H + 1e-60)
        return np.vstack([dy1, dy2])

    def solve(self, x_axis_nm: np.ndarray, boundary_conditions: bc.BoundaryConditions):
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            boundary_conditions.func,
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_{boundary_conditions.get_name()}')

        bf_c, bf_a, bf_s, denom = self.bfactors(sol.y)
        n_cat = self.n_0 * bf_c / denom
        n_an  = self.n_0 * bf_a / denom
        n_sol = self.n_s_0 * bf_s / denom

        # Return solution struct
        ret = prf.SpatialProfilesSolution(
            x=sol.x,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_cat=n_cat/1e3/C.N_A,
            c_an=n_an/1e3/C.N_A,
            c_sol=n_sol/1e3/C.N_A,
            eps=rel_permittivity(self.eps_r_opt, self.n_0, n_sol, self.p_tilde, sol.y[1, :]),
            name=self.name)
        return ret