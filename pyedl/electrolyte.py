"""
Utility functions for double-layer models
"""

from typing import List, Union
from dataclasses import dataclass, field
from scipy import constants
import numpy as np

from . import defaults as D


def calculate_dipmom(
    min_eps: float, max_eps: float, temperature: float, concentration: float
) -> float:
    """
    Calculate the effective dipole moment of a solvent molecule based on its measured
    permittivity, in elementary charge/Angstrom
    """
    dipmom = np.sqrt(
        3
        * (max_eps - min_eps)
        * constants.epsilon_0
        * constants.Boltzmann
        * temperature
        / (concentration * 1e3 * constants.Avogadro)
    )
    return dipmom / constants.elementary_charge / constants.angstrom


@dataclass
class Ion:
    """
    Ion species dataclass
    """

    name: str
    size: float  # relative to smallest species, which should have size 1
    concentration: float  # molar
    charge: float  # units of elementary charge

    def __str__(self):
        return f"Ion: {self.name}, Size={self.size}, Concentration={self.concentration} mol/L, Charge={self.charge}e"

    def __repr__(self):
        return f"Ion(name='{self.name}', size={self.size}, concentration={self.concentration}, charge={self.charge})"


@dataclass
class Solvent:
    """
    Solvent species dataclass
    """

    name: str
    size: float  # relative to smallest species, which should have size 1
    concentration: float  # molar
    min_eps: float  # relative to vacuum permittivity
    dipole_moment: float  # e·A

    def __str__(self):
        return f"{self.name}: Size={self.size}, Concentration={self.concentration} mol/L, Optical Eps={self.min_eps}·eps0, Dipole Moment={self.dipole_moment} e·Å"

    def __repr__(self):
        return f"Solvent(name='{self.name}', size={self.size}, concentration={self.concentration}, min_eps={self.min_eps}, dipole_moment={self.dipole_moment})"


@dataclass
class Water(Solvent):
    """
    Default water parameters
    """

    name: str = "H2O"
    size: float = 1.0
    concentration: float = D.WATER_BULK_M
    min_eps: float = D.WATER_REL_ELEC_EPS
    dipole_moment: float = field(init=False)

    def __post_init__(self):
        # Initialize or calculate the dipole moment based on the constants
        self.dipole_moment = calculate_dipmom(
            min_eps=self.min_eps,
            max_eps=D.WATER_REL_EPS,
            temperature=D.DEFAULT_TEMPERATURE,
            concentration=self.concentration,
        )


Species = Union[Ion, Solvent]


class LatticeElectrolyte:
    """
    Class for specifying the electrolyte species using the lattice gas description
    """

    def __init__(self, species: List[Species]):
        self.species = species
        self._account_for_decrement()
        self.n_site = np.sum(self.ion_sizes * self.ion_n_b) + np.sum(
            self.sol_sizes * self.sol_n_b
        )  # lattice site density

    def _account_for_decrement(self) -> None:
        """
        Update solvent bulk concentrations to account for the reduction in free solvent
        concentration due to space occupied by (solvated) ions
        """
        decrement = 1 - np.sum(self.ion_n_b * self.ion_sizes) / np.sum(
            self.sol_n_b * self.sol_sizes
        )

        for spec in self.species:
            if isinstance(spec, Solvent):
                spec.concentration = spec.concentration * decrement

    def get_properties(self, species_type: type, property_name: str) -> np.ndarray:
        """
        Get properties as numpy array for a certain type of species
        """
        return np.array(
            [
                getattr(s, property_name)
                for s in self.species
                if isinstance(s, species_type)
            ]
        )

    @property
    def ion_concentrations(self):
        return self.get_properties(Ion, "concentration")

    @property
    def ion_sizes(self):
        return self.get_properties(Ion, "size")

    @property
    def ion_q(self):
        return self.get_properties(Ion, "charge")

    @property
    def ion_n_b(self):
        """Bulk number density"""
        return (
            self.get_properties(Ion, "concentration")
            * 1e3
            * constants.Avogadro
            * constants.angstrom**3
        )

    @property
    def ion_f_b(self):
        """Bulk occupied volume fraction"""
        return self.ion_sizes * self.ion_n_b / self.n_site

    @property
    def sol_concentrations(self):
        return self.get_properties(Solvent, "concentration")

    @property
    def sol_sizes(self):
        return self.get_properties(Solvent, "size")

    @property
    def sol_p(self):
        return self.get_properties(Solvent, "dipole_moment")

    @property
    def sol_n_b(self):
        """Bulk number density"""
        return (
            self.get_properties(Solvent, "concentration")
            * 1e3
            * constants.Avogadro
            * constants.angstrom**3
        )

    @property
    def sol_f_b(self):
        """Bulk occupied volume fraction"""
        return self.sol_sizes * self.sol_n_b / self.n_site

    @property
    def min_eps(self):
        """Optical permittivity"""
        return np.sum(self.sol_n_b * self.get_properties(Solvent, "min_eps")) / np.sum(
            self.sol_n_b
        )

    def __str__(self):
        return "\n".join(str(s) for s in self.species)

    def __repr__(self):
        return f"Electrolyte(species={self.species})"
