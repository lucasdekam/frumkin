<<<<<<< HEAD
import constants as C
import numpy as np
from scipy.integrate import solve_bvp
from abc import ABC, abstractmethod
import os
import pickle
import pandas as pd
import multiprocessing as mp
from dataclasses import dataclass

@dataclass
class DoubleLayerModelSolution:
    x:      np.ndarray  # position, nm
    phi:    np.ndarray  # potential, V
    efield: np.ndarray  # electric field, V/m
    c_cat:  np.ndarray  # cation concentration, M
    c_an:   np.ndarray  # anion concentration, M
    c_sol:  np.ndarray  # solvent concentration, M
    eps:    np.ndarray  # relative permittivity

@dataclass
class PotentialSweepSolution:
    phi:    np.ndarray  # potential, V
    charge: np.ndarray  # surface charge, C/m^2
    cap:    np.ndarray  # differential capacity, uF/cm2
=======
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


def get_odesol(x_axis, ode_rhs, boundary_condition, model_name, spec_string, verbose=True, force_recalculation=True):
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
            ode_rhs,
            boundary_condition,
            x_axis,
            y,
            max_nodes=1000000,
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
>>>>>>> parent of 8bf8884 (Merge pull request #2 from lucasdekam/fix-dipolarpb)


class DoubleLayerModel(ABC):
    """
<<<<<<< HEAD
    Abstract base class for a double layer model, requiring all 
    derived classes to have the same methods
    """
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def getPotentialSweepSolution(self, potential_V):
        pass

    @abstractmethod
    def getDoubleLayerModelSolution(self, x_nm, phi0_V):
        pass


class GuyChapman(DoubleLayerModel):
    """
    Guy-Chapman model, treating ions as point particles obeying Boltzmann statistics.

    See e.g. Schmickler & Santos' Interfacial Electrochemistry.
    """
    def __init__(self, ionconc_M, pzc_V):
        self.n_0 = ionconc_M * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.z*C.e_0)**2/(C.eps_r_water*C.eps_0*C.k_B*C.T))
        self.pzc_V = pzc_V

    def getPotentialSweepSolution(self, potential_V):
        cap = self.kappa_debye * C.eps_r_water * C.eps_0 * np.cosh(C.beta*C.z*C.e_0/2 * (potential_V - self.pzc_V)) * 1e2
        ret = PotentialSweepSolution(phi=potential_V, charge=np.zeros(potential_V.shape), cap=cap)
        return ret

    def getDoubleLayerModelSolution(self, x_nm, phi0_V):
        return None


class BorukhovAndelmanOrland(DoubleLayerModel):
    """
    Model developed by Borukhov, Andelman and Orland, modifying the Guy-Chapman model to 
    take finite ion size into account.

    https://doi.org/10.1016/S0013-4686(00)00576-4
    """
    def __init__(self, ionconc_M, pzc_V, d_ions_m):
        self.n_0 = ionconc_M * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.z*C.e_0)**2/(C.eps_r_water*C.eps_0*C.k_B*C.T))
        self.pzc_V = pzc_V

        self.a = d_ions_m
        self.chi0 = 2 * d_ions_m ** 3 * self.n_0 

    def capacitanceIntermediateStep_uFcm2(self, phi_V):
        return np.sqrt(2) * self.kappa_debye * C.eps_r_water * C.eps_0 / np.sqrt(self.chi0) \
            * self.chi0 * np.sinh(C.beta*C.z*C.e_0*phi_V) \
            / (self.chi0 * np.cosh(C.beta*C.z*C.e_0*phi_V) - self.chi0 + 1) \
            / (2*np.sqrt(np.log(self.chi0 * np.cosh(C.beta*C.z*C.e_0*phi_V) - self.chi0 + 1))) \
            * 1e2 # uF/cm^2

    def getPotentialSweepSolution(self, potential_V):
        # cap = self.capacitanceIntermediateStep_uFcm2(potential_V - self.pzc_V) * (potential_V > self.pzc_V) \
        #     - self.capacitanceIntermediateStep_uFcm2(potential_V - self.pzc_V) * (potential_V <= self.pzc_V)
        cap = np.sqrt(2) * self.kappa_debye * C.eps_r_water * C.eps_0 / np.sqrt(self.chi0) \
            * self.chi0 * np.sinh(C.beta*C.z*C.e_0*np.abs(potential_V)) \
            / (self.chi0 * np.cosh(C.beta*C.z*C.e_0*potential_V) - self.chi0 + 1) \
            / (2*np.sqrt(np.log(self.chi0 * np.cosh(C.beta*C.z*C.e_0*potential_V) - self.chi0 + 1))) \
            * 1e2 # uF/cm^2
        ret = PotentialSweepSolution(phi=potential_V, charge=np.zeros(potential_V.shape), cap=cap)
        return ret


    def getDoubleLayerModelSolution(self, x_nm, phi0_V):
        return None


class Huang(DoubleLayerModel):
    """
    Model developed by Jun Huang and co-workers, taking into account finite ion size and 
    dipole moments of the solution molecules.

    https://doi.org/10.1021/acs.jctc.1c00098
    https://doi.org/10.1021/jacsau.1c00315
    """
    def __init__(self, ionconc_M, pzc_V, d_cation_m, d_anion_m, model_water_molecules=True):
        self.n_0 = ionconc_M * 1e3 * C.N_A
        self.pzc_V = pzc_V
        self.model_water_molecules = model_water_molecules

        n_water_bulk = C.c_water_bulk * C.N_A
        self.n_max = n_water_bulk + 2 * self.n_0 

        d_solvent_m = (1/self.n_max)**(1/3)  # water molecule diameter, m
        self.dc = d_cation_m 
        self.da = d_anion_m
        self.gamma_c = d_cation_m**3/d_solvent_m**3
        self.gamma_a = d_anion_m**3/d_solvent_m**3 
        self.chi = self.n_0 / self.n_max

        self.kappa = np.sqrt(2*self.n_0*(C.z*C.e_0)**2/(C.eps_0*C.eps_r_water*C.k_B*C.T))
        self.eps_r_opt = 1
        if not self.model_water_molecules:
            self.eps_r_opt = C.eps_r_water
        self.p = np.sqrt(3 * C.k_B * C.T * (C.eps_r_water - self.eps_r_opt) * C.eps_0 / n_water_bulk)
        self.ptilde = self.p * self.kappa / (C.z * C.e_0)

    def computeBoltzmannFactorsAndOmega(self, y):
        """
        Compute the Boltzmann factors
        BFc = exp(-z e beta phi)
        BFa = exp(+z e beta phi)
        BFs = sinh(beta p E)/(beta p E)

        minimum and maximum are to avoid infinities or division by zero
        """
        BFc = np.minimum(np.exp(-y[0, :]), 1e60)
        BFa = np.minimum(np.exp(+y[0, :]), 1e60)
        BFs = None
        if self.model_water_molecules:
            BFs = np.maximum(np.minimum(np.sinh(self.ptilde * y[1, :]), 1e60)/(self.ptilde * y[1, :] + 1e-60), 1e-60)
        else:
            BFs = 1 # If we don't model the water molecule dipoles, p=0 so sinh x/x = 1
        Omega = (1 - 2*self.chi) * BFs + self.gamma_c * self.chi * BFc + self.gamma_a * self.chi * BFa

        return BFc, BFa, BFs, Omega

    def langevinOfXOverX(self, x):
        """
        Returns L(x)/x, where L(x) is the Langevin function:
        L(x) = 1/tanh(x) - 1/x
        For small x, the function value is 1/3
        """
        ret = np.zeros(np.atleast_1d(x).shape)
        ret = (1/np.tanh(x) - 1/x)/x
        ret[np.abs(x) <= 1e-9] = 1/3
        return ret

    def computePermittivity(self, BFa, BFc, Omega, soly_1):
        """
        Compute the permittivity using the electric field
        """
        eps = None
        if self.model_water_molecules:
            eps = self.eps_r_opt + 1/2 * C.eps_r_water * self.ptilde**2 * (1 - self.chi * BFc / Omega - self.chi * BFa/Omega) * self.langevinOfXOverX(self.ptilde * soly_1) / self.chi
        else:
            eps = np.ones(soly_1.shape) * C.eps_r_water
        return eps
    
    def odeSystem(self, x, y):
        """
        System of dimensionless 1st order ODE's that we solve

        x: dimensionless x-axis of length n, i.e. kappa (1/m) times x-position (m).
        y: dimensionless potential phi and dphi/dx, size 2 x n.
        """        
        dy1 = y[1, :]

        BFc, BFa, BFs, Omega = self.computeBoltzmannFactorsAndOmega(y)        
        H = 1/2 * (C.eps_r_water / self.eps_r_opt) * (1 - 1/BFs**2) * (1 - self.chi * BFc / Omega - self.chi * BFa/Omega) / self.chi

        dy2 = None
        if self.model_water_molecules:
            dy2 = - 1/2 * (C.eps_r_water / self.eps_r_opt) * (BFc - BFa) / Omega * y[1, :]**2 / (y[1, :]**2 + H + 1e-60)
        else:
            dy2 = - 1/2 * (C.eps_r_water / self.eps_r_opt) * (BFc - BFa) / Omega
        
        return np.vstack([dy1, dy2])

    def bc(self, phi_bc_V):
        """
        Boundary conditions: fixed potential at the metal, zero potential at "infinity" (or: far enough away)

        Returns a boundary condition function for use in scipy's solve_bvp
        """
        return lambda ya, yb : np.array([ya[0] - phi_bc_V * C.beta * C.z * C.e_0, yb[0]])   

    def getXAxis_m(self, xmax_m, N):
        """
        Get a logarithmically spaced x-axis, fine mesh close to electrode
        """
        xmax_nm = xmax_m * 1e9
        expmax = np.log10(xmax_nm)
        x = np.logspace(-9, expmax, N) - 1e-9
        return x*1e-9

    def getOdeSol(self, xmax_m, N, phi0_V, verbose=False, force_recalculation=False):
        """
        Solve the ODE system or load the solution if there is one (if we want to plot many things)
        """
        x = self.kappa * self.getXAxis_m(xmax_m, N) # dimensionless x-axis
        y = np.zeros((2, x.shape[0])) 

        sol = None 

        pickle_name = f'sol_huang__c0__{self.n_0/C.N_A/1e3:.3f}__xmax_{xmax_m*1e9:.0f}nm__N_{N}__phi0_{phi0_V:.2f}__dc_{self.dc*1e10:.0f}__da_{self.da*1e10:.0f}.pkl'
        folder_name = './solutions/'
        if pickle_name not in os.listdir(folder_name) or force_recalculation:
            sol = solve_bvp(self.odeSystem, self.bc(phi0_V), x, y, max_nodes=1000000, verbose=verbose)
            pickle.dump(sol, open(folder_name+pickle_name, 'wb'))
            if verbose:
                print(f'Solved and saved under {folder_name+pickle_name}.')
        else:
            if verbose:
                print(f'File {pickle_name} found.')
            sol = pickle.load(open(folder_name+pickle_name, 'rb'))

        return sol

    def getDoubleLayerModelSolution(self, xmax_m, N, phi0_V, force_recalculation=False):
        sol = self.getOdeSol(xmax_m, N, phi0_V, verbose=True, force_recalculation=force_recalculation)
        BFc, BFa, _, Omega = self.computeBoltzmannFactorsAndOmega(sol.y)
        
        c_cat = self.n_max * self.chi * BFc / Omega / C.N_A / 1e3
        c_an = self.n_max * self.chi * BFa / Omega / C.N_A / 1e3
        c_sol = self.n_max / C.N_A / 1e3 - c_cat - c_an

        eps = self.computePermittivity(BFa, BFc, Omega, sol.y[1, :])
        ret = DoubleLayerModelSolution(
            x=sol.x / self.kappa * 1e9, 
            phi=sol.y[0, :] / (C.beta * C.z * C.e_0),
            efield=sol.y[1, :] * self.kappa / (C.beta * C.z * C.e_0), 
            c_cat=c_cat,
            c_an=c_an,
            c_sol=c_sol,
            eps=eps)
        return ret

    def computeCharge(self, xmax_m, N, potential_V, force_recalculation=False):
        """
        Compute the surface charge at a given potential
        """
        sol = self.getOdeSol(xmax_m, N, potential_V, verbose=False, force_recalculation=force_recalculation)   

        BFc, BFa, _, Omega = self.computeBoltzmannFactorsAndOmega(sol.y)
        eps = self.computePermittivity(BFa, BFc, Omega, sol.y[1, :])
        dphidx = - sol.y[1, :] * self.kappa / (C.beta * C.z * C.e_0)

        charge_C = C.eps_0 * eps[0] * dphidx[0]
        return charge_C

    def getPotentialSweepSolution(self, potential_V, force_recalculation=False):
        """
        Compute the differential capacitance in microfarads per square cm. 
        Parallelized using the multiprocessing Python module.
        """
        xmax_m = 100e-9
        N = 10000

        pool = mp.Pool(mp.cpu_count())
        charge = pool.starmap(self.computeCharge, [(xmax_m, N, phi, force_recalculation) for phi in potential_V])
        pool.close()

        cap = np.gradient(charge, edge_order=2) / np.gradient(potential_V) * 1e2
        ret = PotentialSweepSolution(phi=potential_V, charge=charge, cap=cap)
        return ret
=======
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
    def solve(self,
            x_axis_nm: np.ndarray,
            boundary_conditions: bc.BoundaryConditions,
            force_recalculation=True):
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

    def solve(self,
            x_axis_nm: np.ndarray,
            boundary_conditions: bc.BoundaryConditions,
            force_recalculation=True):
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            boundary_conditions.func,
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_{boundary_conditions.get_name()}',
            force_recalculation=force_recalculation)

        # Return solution struct
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
        self.name = 'Borukhov'

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        dy2 = np.sinh(y[0, :]) / (1 - self.chi_0 + self.chi_0 * np.cosh(y[0, :]))
        return np.vstack([dy1, dy2])

    def solve(self,
            x_axis_nm: np.ndarray,
            boundary_conditions: bc.BoundaryConditions,
            force_recalculation=True):
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            boundary_conditions.func,
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_{boundary_conditions.get_name()}',
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

    def solve(self,
            x_axis_nm: np.ndarray,
            boundary_conditions: bc.BoundaryConditions,
            force_recalculation=True):
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            boundary_conditions.func,
            self.name,
            f'c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_{boundary_conditions.get_name()}',
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
            eps=rel_permittivity(self.eps_r_opt, self.n_0, n_sol, self.p_tilde, sol.y[1, :]),
            name=self.name)
        return ret


class Huang(Abrashkin):
    """
    Adapation of Huang, Chen and Eikerling's extension of Abrashkin's
    model to take into account different solvent molecule and ion sizes.
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
        d_sol_m:                    solvent molecule diameter in meters; lattice spacing
        eps_r_opt:                  optical/background relative permittivity, default: 1
        """
        self.c_0 = ion_concentration_molar
        self.n_0 = self.c_0 * 1e3 * C.N_A
        self.n_max = 1 / d_sol_m ** 3
        self.kappa_debye = np.sqrt(2*self.n_0*(C.Z*C.E_0)**2 /
                                   (C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))

        self.gamma_c = (d_cat_m / d_sol_m) ** 3
        self.gamma_a = (d_an_m / d_sol_m) ** 3
        self.n_s_0 = self.n_max - self.n_0 - self.n_0

        self.chi = self.n_0 / self.n_max
        self.chi_s = self.n_s_0 / self.n_max

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
        denom = self.chi_s*bf_s + self.gamma_c*self.chi*bf_c + self.gamma_a*self.chi*bf_a
        n_cat = self.n_max * self.chi * bf_c / denom
        n_an  = self.n_max * self.chi * bf_a / denom
        n_sol = self.n_max - n_cat - n_an  # weird!
        # n_sol = self.n_max - self.gamma_c * n_cat - self.gamma_a * n_an
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

        self.n_max = self.n_s_0 + 2 * self.n_0 ## weird...
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
        n_sol = self.n_max - n_cat - n_an # weird!!!
        return n_cat, n_an, n_sol


class Species:
    """
    Keeps track of concentration of a species in the Multispecies double layer model,
    and how to calculate its Boltzmann factor
    """
    def __init__(self, concentration_molar: float,
                       diameter_m: float,
                       charge: float,
                       name: str) -> None:
        self.c_0 = concentration_molar
        self.n_0 = concentration_molar * 1e3 * C.N_A
        self.charge = charge
        self.diameter_m = diameter_m
        self.name = name

        self.chi = None
        self.gamma = 1

    def bfac(self, y_0):
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
    def __init__(self, diameter_m: float, name: str) -> None:
        n_max = 1/diameter_m**3
        c_max = n_max/1e3/C.N_A
        super().__init__(c_max, diameter_m, 0, name)

    def bfac(self, p_tilde: float, y_1: np.ndarray):
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

        charge_densities = [s.charge * s.c_0 for s in self.species]
        if sum(charge_densities) > 1e-9:
            raise ValueError('No charge neutrality')

        ionic_densities = [s.charge**2 * s.n_0 for s in self.species]
        self.ionic_str = sum(ionic_densities) / 2
        self.kappa_debye = np.sqrt(2*self.ionic_str*C.E_0**2 /
                                   (C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))

        self.n_max = 1/self.solvent.diameter_m**3  ## lattice spacing: d_s

        for species in self.species:
            species.chi = species.n_0 / self.n_max
            species.gamma = (species.diameter_m / self.solvent.diameter_m) ** 3
            solvent.n_0 -= species.n_0

        solvent.c_0 = solvent.n_0/1e3/C.N_A
        solvent.chi = self.solvent.n_0 / self.n_max

        # Computing the dipole moment p and the dimensionless ptilde
        p_water = np.sqrt(3 * C.K_B * C.T * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / self.solvent.n_0)
        self.p_tilde = p_water * self.kappa_debye / (C.Z * C.E_0)

        self.name = f'Multispecies {self.ionic_str/1e3/C.N_A:.4f}'

    def density_denominator(self, sol_y):
        """
        Compute the denominator appearing in concentration or density expressions:
        chi_v + chi_sol + sum of chi_i
        """
        boltzmann_chi_gamma = [s.gamma * s.chi * s.bfac(sol_y[0, :]) for s in self.species]
        sol_bfac = self.solvent.bfac(self.p_tilde, sol_y[1, :])
        denom = self.solvent.gamma * self.solvent.chi * sol_bfac + sum(boltzmann_chi_gamma)
        return denom

    def concentration_dict(self, sol_y, denom):
        """
        Compute concentration profiles of the different species and the solvent, and return
        those as a dict.
        """
        c_dict = {s.name: s.n_0 * s.bfac(sol_y[0, :])/denom/1e3/C.N_A for s in self.species}
        c_list = [value for key, value in c_dict.items()]
        c_dict['Solvent'] = self.n_max/1e3/C.N_A - sum(c_list) ## Weird!!!
        return c_dict

    def charge_density_list(self, sol_y, denom):
        """
        Returns a list of the charge densities of all (charged) species, in units of
        the elementary charge C.E_0.
        """
        rho_list = [s.charge * s.n_0 * s.bfac(sol_y[0, :])/denom for s in self.species]
        return rho_list

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        denom = self.density_denominator(y)
        rho_list = self.charge_density_list(y, denom)
        n_sol = self.concentration_dict(y, denom)['Solvent']*1e3*C.N_A
        sol_bfac = self.solvent.bfac(self.p_tilde, y[1, :])

        H = 1/2 * C.EPS_R_WATER/self.eps_r_opt * (1-1/sol_bfac**2) * n_sol/self.ionic_str
        dy2 = - 1/2 * (C.EPS_R_WATER/self.eps_r_opt) * sum(rho_list)/self.ionic_str * y[1, :]**2 / (y[1, :]**2 + H + 1e-60)
        return np.vstack([dy1, dy2])

    def solve(self,
            x_axis_nm: np.ndarray,
            boundary_conditions: bc.BoundaryConditions,
            force_recalculation=True):
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            boundary_conditions.func,
            self.name,
            f'c0_{self.ionic_str/1e3/C.N_A:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__bc_{boundary_conditions.get_name()}',
            force_recalculation=force_recalculation)

        denom = self.density_denominator(sol.y)
        c_dict = self.concentration_dict(sol.y, denom)
        eps = rel_permittivity(
            self.eps_r_opt,
            self.ionic_str,
            c_dict['Solvent']*1e3*C.N_A,
            self.p_tilde,
            sol.y[1, :])

        ret = prf.SpatialProfilesSolution(
            x=sol.x / self.kappa_debye * 1e9,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_dict=c_dict,
            eps=eps,
            name=self.name)

        return ret


class InsulatorBorukhov(Borukhov):
    """
    Class for modeling insulators
    """
    def __init__(self, p_h: float, support_molar: float, a_m: float):
        self.p_h_bulk = p_h
        self.p_oh_bulk = C.PKW - p_h
        self.proton_bulk_molar = 10 ** (-self.p_h_bulk)
        self.oh_bulk_molar = 10 ** (-self.p_oh_bulk)
        super().__init__(support_molar + self.proton_bulk_molar + self.oh_bulk_molar, a_m)

    def insulator_boundary_condition(self, ya, yb):
        """
        Robin boundary condition for the insulator
        """
        h_surf = self.proton_bulk_molar * np.exp(-ya[0]) / (1 - self.chi_0 + self.chi_0 * np.cosh(ya[0]))
        h_surf = max(h_surf.squeeze(), 0)

        # left = 2 * ya[1] - C.N_SITES_SILICA * self.kappa_debye *C.K_SILICA_A / (C.K_SILICA_A + h_surf) / self.n_0
        left = 2 * self.n_0 / self.kappa_debye * ya[1] \
            + C.N_SITES_SILICA * (h_surf**2 - C.K_SILICA_A * C.K_SILICA_B) \
            / (C.K_SILICA_A * C.K_SILICA_B + C.K_SILICA_B * h_surf + h_surf ** 2)
        right = yb[0]

        return np.array([left, right])

    def solve_ins(self,
            x_axis_nm: np.ndarray,
            verbose=2,
            force_recalculation=True):
        """
        Solve using insulator BC
        """
        # Obtain potential and electric field
        x_axis = self.kappa_debye * 1e-9 * x_axis_nm
        sol = get_odesol(
            x_axis,
            self.ode_rhs,
            self.insulator_boundary_condition,
            self.name,
            f'ins__c0_{self.c_0:.4f}M__xmax_{x_axis_nm[-1]:.0f}nm__ph_{self.p_h_bulk:.2f}',
            verbose=verbose,
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
            eps=np.ones(sol.x.shape) * C.EPS_R_WATER,
            name=self.name)
        return ret
>>>>>>> parent of 8bf8884 (Merge pull request #2 from lucasdekam/fix-dipolarpb)
