"""
Defining electrolytes for double-layer models.
"""

from typing import List, Union
from dataclasses import dataclass, field
from scipy import constants
import numpy as np

from .tools import defaults as D


def calculate_dipmom(
    min_eps: float, max_eps: float, temperature: float, concentration: float
) -> float:
    """
    Calculate the effective dipole moment of a solvent molecule based on its measured
    permittivity, in elementary charge/angstrom.

    Parameters
    ----------
    min_eps : float
        Minimum permittivity (optical permittivity).
    max_eps : float
        Maximum permittivity (static permittivity).
    temperature : float
        Temperature in Kelvin.
    concentration : float
        Solvent concentration in mol/L.

    Returns
    -------
    float
        Effective dipole moment in e·Å.
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
    Represents an ion species.

    Attributes
    ----------
    name : str
        Name of the ion.
    size : float
        Size of the ion relative to the lattice sites. Usually, the lattice sites
        have the size of the smallest species (which should then have size 1).
    concentration : float
        Molar concentration of the ion.
    charge : float
        Charge of the ion in units of elementary charge.
    """

    name: str
    size: float
    concentration: float
    charge: float

    def __str__(self):
        return f"Ion: {self.name}, Size={self.size}, Concentration={self.concentration} mol/L, Charge={self.charge}e"

    def __repr__(self):
        return f"Ion(name='{self.name}', size={self.size}, concentration={self.concentration}, charge={self.charge})"


@dataclass
class Solvent:
    """
    Represents a solvent species.

    Attributes
    ----------
    name : str
        Name of the solvent.
    size : float
        Size of the solvent relative to the lattice sites. Usually, the lattice sites
        have the size of the smallest species (which should then have size 1).
    concentration : float
        Molar concentration of the solvent.
    min_eps : float
        Minimum permittivity (optical permittivity) relative to vacuum permittivity.
    dipole_moment : float
        Dipole moment of the solvent in e·Å.
    """

    name: str
    size: float
    concentration: float
    min_eps: float
    dipole_moment: float

    def __str__(self):
        return f"{self.name}: Size={self.size}, Concentration={self.concentration} mol/L, Optical Eps={self.min_eps}·eps0, Dipole Moment={self.dipole_moment} e·Å"

    def __repr__(self):
        return f"Solvent(name='{self.name}', size={self.size}, concentration={self.concentration}, min_eps={self.min_eps}, dipole_moment={self.dipole_moment})"


@dataclass
class Water(Solvent):
    """
    Represents water with default parameters.

    Attributes
    ----------
    name : str
        Name of the solvent, default is 'H2O'.
    size : float
        Size of the water molecule, default is 1.0.
    concentration : float
        Molar concentration of water, default is defined in defaults.
    min_eps : float
        Minimum permittivity (optical permittivity) relative to vacuum permittivity, default is defined in defaults.
    dipole_moment : float
        Dipole moment of water, calculated during initialization.
    """

    name: str = r"H$_2$O"
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
    Specifies the electrolyte species using the lattice gas description.

    Attributes
    ----------
    species : List[Species]
        List of species (ions and solvents) in the electrolyte.
    n_site : float
        Lattice site density.
    """

    def __init__(self, species: List[Species]):
        self.species = species
        self._account_for_decrement()

    def _account_for_decrement(self) -> None:
        """
        Update solvent bulk concentrations to account for the reduction in free solvent
        concentration due to space occupied by (solvated) ions.
        """
        decrement = 1 - np.sum(self.ion_n_b * self.ion_sizes) / np.sum(
            self.sol_n_b * self.sol_sizes
        )

        for spec in self.species:
            if isinstance(spec, Solvent):
                spec.concentration *= decrement

    def get_properties(self, species_type: type, property_name: str) -> np.ndarray:
        """
        Get properties as a numpy array for a certain type of species.

        Parameters
        ----------
        species_type : type
            The class type (Ion or Solvent).
        property_name : str
            The name of the property to retrieve.

        Returns
        -------
        np.ndarray
            Array of property values.
        """
        return np.array(
            [
                getattr(s, property_name)
                for s in self.species
                if isinstance(s, species_type)
            ]
        )

    def ohp(self, surf_pot: float) -> float:
        """
        Get the location of the outer Helmholtz plane based on the electric potential
        at the surface. Negative surface potentials attract positive counterions, so
        the ohp depends on the size of the smallest positive ion, and vice versa.

        Parameters
        ----------
        surf_pot : float
            Surface potential (any unit, only the sign matters).

        Returns
        -------
        float
            Distance to the outer Helmholtz plane.
        """
        counterions = self.ion_q * surf_pot <= 0
        ohp = 1 / 2 * (self.ion_sizes[counterions] / self.n_site) ** (1 / 3)
        return min(ohp)

    @property
    def n_site(self) -> float:
        """Lattice site density."""
        return np.sum(self.ion_sizes * self.ion_n_b) + np.sum(
            self.sol_sizes * self.sol_n_b
        )

    @property
    def ion_concentrations(self) -> np.ndarray:
        """Ion concentrations in mol/L."""
        return self.get_properties(Ion, "concentration")

    @property
    def ion_sizes(self) -> np.ndarray:
        """Sizes of ions relative to the smallest species."""
        return self.get_properties(Ion, "size")

    @property
    def ion_q(self) -> np.ndarray:
        """Ion charges in units of elementary charge."""
        return self.get_properties(Ion, "charge")

    @property
    def ion_n_b(self) -> np.ndarray:
        """Bulk number density of ions."""
        return (
            self.ion_concentrations
            / constants.deci**3
            * constants.Avogadro
            * constants.angstrom**3
        )

    @property
    def ion_f_b(self) -> np.ndarray:
        """Bulk occupied volume fraction of ions."""
        return self.ion_sizes * self.ion_n_b / self.n_site

    @property
    def ion_names(self) -> list:
        """Ion names"""
        return [s.name for s in self.species if isinstance(s, Ion)]

    @property
    def sol_concentrations(self) -> np.ndarray:
        """Solvent concentrations in mol/L."""
        return self.get_properties(Solvent, "concentration")

    @property
    def sol_sizes(self) -> np.ndarray:
        """Sizes of solvents relative to the smallest species."""
        return self.get_properties(Solvent, "size")

    @property
    def sol_p(self) -> np.ndarray:
        """Dipole moments of solvents in e·Å."""
        return self.get_properties(Solvent, "dipole_moment")

    @property
    def sol_n_b(self) -> np.ndarray:
        """Bulk number density of solvents."""
        return (
            self.sol_concentrations
            / constants.deci**3
            * constants.Avogadro
            * constants.angstrom**3
        )

    @property
    def sol_f_b(self) -> np.ndarray:
        """Bulk occupied volume fraction of solvents."""
        return self.sol_sizes * self.sol_n_b / self.n_site

    @property
    def sol_names(self) -> list:
        """Solvent names"""
        return [s.name for s in self.species if isinstance(s, Solvent)]

    @property
    def min_eps(self) -> float:
        """Optical permittivity of the solvent."""
        return np.sum(self.sol_n_b * self.get_properties(Solvent, "min_eps")) / np.sum(
            self.sol_n_b
        )

    def __str__(self):
        return "\n".join(str(s) for s in self.species)

    def __repr__(self):
        return f"Electrolyte(species={self.species})"
