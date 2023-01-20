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


def get_odesol(x_axis, odefunc, boundary_condition, model_name, spec_string, verbose=True, force_recalculation=True):
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
    if pickle_name not in os.listdir(folder_path) or force_recalculation:
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


class Abrashkin(DoubleLayerModel):
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

    def solvent_factor(self, y_1: np.ndarray):
        """
        Compute solvent Boltzmann factor.
        y_1: dimensionless electric field
        """
        bf_s = np.zeros(y_1.shape)
        select = np.abs(self.p_tilde * y_1) > 1e-9
        bf_s[select] = np.sinh(self.p_tilde * y_1[select])/(self.p_tilde * y_1[select])
        bf_s[~select] = 1
        return bf_s

    def densities(self, sol_y):
        """
        Compute cation, anion and solvent densities.
        """
        bf_c = np.exp(-sol_y[0, :])
        bf_a = np.exp(+sol_y[0, :])
        bf_s = self.solvent_factor(sol_y[1, :])
        denom = (1 - 2* self.chi) * bf_s + self.chi * (bf_c + bf_a)
        n_cat = self.n_0 * bf_c / denom
        n_an  = self.n_0 * bf_a / denom
        n_sol = self.n_s_0 * bf_s / denom
        return n_cat, n_an, n_sol

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        n_cat, n_an, n_sol = self.densities(y)
        solfac = self.solvent_factor(y[1, :])

        H = 1/2 * C.EPS_R_WATER/self.eps_r_opt * (1-1/solfac**2) * n_sol/self.n_0
        dy2 = - 1/2 * (C.EPS_R_WATER/self.eps_r_opt) * (n_cat - n_an)/self.n_0 * y[1, :]**2 / (y[1, :]**2 + H + 1e-60)
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

        n_cat, n_an, n_sol = self.densities(sol.y)

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


class Huang(Abrashkin):
    """
    Huang, Chen and Eikerling's extension of Abrashkin's model
    to take into account different solvent molecule and ion sizes.
    https://doi.org/10.1021/acs.jctc.1c00098

    Inherits solvent_factor, ode_rhs, and solve from Abrashkin
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, ion_concentration_molar: float,
                       d_cat_m: float,
                       d_an_m: float,
                       d_sol_m: float,
                       eps_r_opt: float = 1):
        # pylint: disable=super-init-not-called
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
        self.a_m = C.LATTICE_SPACING # roughly Bohr radius

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

    def densities(self, sol_y):
        """
        Compute cation, anion and solvent densities.
        """
        bf_c = np.exp(-sol_y[0, :])
        bf_a = np.exp(+sol_y[0, :])
        bf_s = self.solvent_factor(sol_y[1, :])
        denom = self.chi_v + self.gamma_s*self.chi_s*bf_s + self.gamma_c*self.chi*bf_c + self.gamma_a*self.chi*bf_a
        n_cat = self.n_0 * bf_c / denom
        n_an  = self.n_0 * bf_a / denom
        n_sol = self.n_s_0 * bf_s / denom
        return n_cat, n_an, n_sol


class HuangSimple(Abrashkin):
    """
    Huang's simplification in 'Cation Overcrowding Effect on the
    Oxygen Evolution Reaction'
    https://doi.org/10.1021/jacsau.1c00315

    Inherits solvent_factor, ode_rhs, and solve from Abrashkin
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, ion_concentration_molar: float,
                       d_cat_m: float,
                       d_an_m: float,
                       eps_r_opt: float = 1):
        # pylint: disable=super-init-not-called
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

    def densities(self, sol_y):
        """
        Compute cation, anion and solvent densities.
        """
        bf_c = np.exp(-sol_y[0, :])
        bf_a = np.exp(+sol_y[0, :])
        bf_s = self.solvent_factor(sol_y[1, :])
        denom = (1 - 2*self.chi)*bf_s + self.gamma_c*self.chi*bf_c + self.gamma_a*self.chi*bf_a
        n_cat = self.n_0 * bf_c / denom
        n_an  = self.n_0 * bf_a / denom
        n_sol = self.n_s_0 * bf_s / denom
        return n_cat, n_an, n_sol


class Species:
    """
    Keeps track of concentration of a species in the Multispecies double layer model,
    and how to calculate its Boltzmann factor
    """
    def __init__(self, concentration_molar: float,
                       diameter_m: float,
                       charge: float) -> None:
        self.n_0 = concentration_molar * 1e3 * C.N_A
        self.charge = charge
        self.chi = self.n_0 * C.LATTICE_SPACING ** 3
        self.gamma = (diameter_m / C.LATTICE_SPACING) ** 3

    def boltzmann_factor(self, y_0):
        """
        Return the Boltzmann factor, depending on the charge and the
        electric potential.
        y_0: dimensionless electric potential
        """
        return np.exp(-self.charge * y_0)


class Solvent(Species):
    """
    Special solvent species class
    """
    def __init__(self, diameter_m: float) -> None:
        super().__init__(C.C_WATER_BULK, diameter_m, 0)

    def boltzmann_factor(self, p_tilde: float, y_1: np.ndarray):
        """
        Compute solvent Boltzmann factor.
        p_tilde: dimensionless dipole moment
        y_1: dimensionless electric field
        """
        # pylint: disable=arguments-differ
        bf_s = np.zeros(y_1.shape)
        select = np.abs(p_tilde * y_1) > 1e-9
        bf_s[select] = np.sinh(p_tilde * y_1[select])/(p_tilde * y_1[select])
        bf_s[~select] = 1
        return bf_s


class Multispecies(DoubleLayerModel):
    """
    Modification of Huang's model to include multiple ion species.
    """
    def __init__(self, species_list, solvent, eps_r_opt) -> None:
        self.species = species_list
        self.solvent = solvent
        self.eps_r_opt = eps_r_opt

        charge_densities = [s.charge * s.n_0 for s in self.species]
        if sum(charge_densities) != 0:
            raise ValueError('No charge neutrality')

        ionic_densities = [s.charge**2 * s.n_0 for s in self.species]
        self.ionic_str = sum(ionic_densities) / 2
        self.kappa_debye = np.sqrt(2*self.ionic_str*C.E_0**2 /
                                   (C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))

        self.chi_v = 1 - self.solvent.gamma * self.solvent.chi
        for s in self.species: #pylint: disable=invalid-name
            self.chi_v -= s.gamma * s.chi

        # Computing the dipole moment p and the dimensionless ptilde
        p_water = np.sqrt(3 * C.K_B * C.T * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / self.solvent.n_0)
        self.p_tilde = p_water * self.kappa_debye / (C.Z * C.E_0)

        self.name = 'Multispecies'

    def densities(self, sol_y):
        """
        Compute cation, anion and solvent densities.
        """
        bf_s = self.solvent.boltzmann_factor(self.p_tilde, sol_y[1, :])
        boltzmann_times_gamma = [s.gamma * s.chi * s.boltzmann_factor(sol_y[0, :]) for s in self.species]
        denom = self.chi_v + self.solvent.gamma * self.solvent.chi * bf_s + sum(boltzmann_times_gamma)

        n_list = [s.n_0 * s.boltzmann_factor(sol_y[0, :])/denom for s in self.species]
        rho_list = [s.charge * s.n_0 * s.boltzmann_factor(sol_y[0, :])/denom for s in self.species]
        n_sol = self.solvent.n_0 * bf_s / denom
        return n_list, rho_list, n_sol

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        _, rho_list, n_sol = self.densities(y)
        solfac = self.solvent.boltzmann_factor(self.p_tilde, y[1, :])

        H = 1/2 * C.EPS_R_WATER/self.eps_r_opt * (1-1/solfac**2) * n_sol/self.ionic_str
        dy2 = - 1/2 * (C.EPS_R_WATER/self.eps_r_opt) * sum(rho_list)/self.ionic_str * y[1, :]**2 / (y[1, :]**2 + H + 1e-60)
        return np.vstack([dy1, dy2])

    def solve(self, x_axis_nm: np.ndarray, boundary_conditions: bc.BoundaryConditions):
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            boundary_conditions.func,
            self.name,
            f'c0_{self.ionic_str/1e3/C.N_A:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_{boundary_conditions.get_name()}')

        n_list, _, n_sol = self.densities(sol.y)

        return sol.x, sol.y[0, :] / (C.BETA * C.Z * C.E_0), [n/1e3/C.N_A for n in n_list], rel_permittivity(self.eps_r_opt, self.ionic_str, n_sol, self.p_tilde, sol.y[1, :])