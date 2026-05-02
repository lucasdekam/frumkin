"""
Defining electrolytes for double-layer models.
"""

from typing import List, Union, Optional
from dataclasses import dataclass, field
from scipy import constants
import numpy as np

from .tools import defaults as D
from .tools import langevin as L


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
        return (
            f"Ion: {self.name}, Size={self.size}, "
            f"Concentration={self.concentration} mol/L, Charge={self.charge}e"
        )

    def __repr__(self):
        return (
            f"Ion(name='{self.name}', size={self.size}, "
            f"concentration={self.concentration}, charge={self.charge})"
        )


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
        return (
            f"{self.name}: Size={self.size}, Concentration={self.concentration} mol/L, "
            f"Optical Eps={self.min_eps}·eps0, Dipole Moment={self.dipole_moment} e·Å"
        )

    def __repr__(self):
        return (
            f"Solvent(name='{self.name}', size={self.size}, "
            f"concentration={self.concentration}, min_eps={self.min_eps}, "
            f"dipole_moment={self.dipole_moment})"
        )


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
        Optical permittivity relative to vacuum permittivity, default is defined
        in defaults.
    dipole_moment : float
        Dipole moment of water, calculated during initialization.
    """

    name: str = "water"
    size: float = 1.0
    concentration: float = D.WATER_BULK_M
    min_eps: float = D.WATER_REL_ELEC_EPS
    dipole_moment: float = field(init=False)

    def __post_init__(self):
        self.dipole_moment = calculate_dipmom(
            min_eps=self.min_eps,
            max_eps=D.WATER_REL_EPS,
            temperature=D.DEFAULT_TEMPERATURE,
            concentration=self.concentration,
        )


Species = Union[Ion, Solvent]


class LatticeElectrolyte:
    """
    Specifies the electrolyte using a lattice gas description.
    Each lattice site is occupied by exactly one ion, solvent molecule, or vacancy.

    Parameters
    ----------
    species : List[Species]
        Ions and solvents in the electrolyte.
    n_site : float, optional
        Lattice site density in Å⁻³. If not given, derived from species concentrations
        assuming the lattice is fully occupied (no vacancies). Provide explicitly for
        ion-only models (e.g. Bikerman), where vacancies fill the remaining space.
    min_eps : float, optional
        Optical (high-frequency) permittivity. If not given, computed as a
        number-density-weighted average over solvents. Required when no solvents
        are present.
    """

    def __init__(
        self,
        species: List[Species],
        n_site: Optional[float] = None,
        min_eps: Optional[float] = None,
    ):
        self.species = species
        self._n_site_override = n_site
        self._min_eps_override = min_eps

    # ── species accessors ──────────────────────────────────────────────────────

    def get_properties(self, species_type: type, property_name: str) -> np.ndarray:
        """Return a property array for all species of a given type."""
        return np.array(
            [
                getattr(s, property_name)
                for s in self.species
                if isinstance(s, species_type)
            ]
        )

    @property
    def ion_concentrations(self) -> np.ndarray:
        """Ion concentrations in mol/L."""
        return self.get_properties(Ion, "concentration")

    @property
    def ion_sizes(self) -> np.ndarray:
        """Ion sizes relative to the smallest lattice site."""
        return self.get_properties(Ion, "size")

    @property
    def ion_q(self) -> np.ndarray:
        """Ion charges in units of elementary charge."""
        return self.get_properties(Ion, "charge")

    @property
    def ion_names(self) -> list:
        """Ion names."""
        return [s.name for s in self.species if isinstance(s, Ion)]

    @property
    def sol_concentrations(self) -> np.ndarray:
        """Solvent concentrations in mol/L (as specified; not reduced for excluded volume)."""
        return self.get_properties(Solvent, "concentration")

    @property
    def sol_sizes(self) -> np.ndarray:
        """Solvent sizes relative to the smallest lattice site."""
        return self.get_properties(Solvent, "size")

    @property
    def sol_dipole(self) -> np.ndarray:
        """Solvent dipole moments in e·Å."""
        return self.get_properties(Solvent, "dipole_moment")

    @property
    def sol_names(self) -> list:
        """Solvent names."""
        return [s.name for s in self.species if isinstance(s, Solvent)]

    # ── bulk densities and volume fractions ───────────────────────────────────

    @property
    def ion_bulk_nd(self) -> np.ndarray:
        """Ion bulk number densities in Å⁻³."""
        return (
            self.ion_concentrations
            / constants.deci**3
            * constants.Avogadro
            * constants.angstrom**3
        )

    @property
    def _sol_bulk_nd_raw(self) -> np.ndarray:
        """Solvent number densities in Å⁻³ directly from species.concentration."""
        return (
            self.sol_concentrations
            / constants.deci**3
            * constants.Avogadro
            * constants.angstrom**3
        )

    @property
    def sol_bulk_nd(self) -> np.ndarray:
        """
        Solvent bulk number densities in Å⁻³, reduced so that ions and solvents
        together fill the lattice (Σ bulk volume fractions = 1). Skipped when
        n_site is provided explicitly, in which case vacancies fill remaining sites.
        """
        raw = self._sol_bulk_nd_raw
        if len(raw) == 0 or self._n_site_override is not None:
            return raw
        # Scale solvent so that: Σ size_i·n_i + Σ size_s·n_s = n_site_pure_solvent
        ion_vol_density = np.sum(self.ion_sizes * self.ion_bulk_nd)
        n_site_pure = np.sum(raw * self.sol_sizes)
        return raw * (n_site_pure - ion_vol_density) / n_site_pure

    @property
    def n_site(self) -> float:
        """Lattice site density in Å⁻³ (= 1/a³ where a is the lattice spacing)."""
        if self._n_site_override is not None:
            return self._n_site_override
        return np.sum(self.ion_sizes * self.ion_bulk_nd) + np.sum(
            self.sol_sizes * self.sol_bulk_nd
        )

    @property
    def ion_bulk_vf(self) -> np.ndarray:
        """Ion bulk volume fractions f_b,i = size_i · n_b,i / n_site."""
        return self.ion_sizes * self.ion_bulk_nd / self.n_site

    @property
    def sol_bulk_vf(self) -> np.ndarray:
        """Solvent bulk volume fractions f_b,s = size_s · n_b,s / n_site."""
        return self.sol_sizes * self.sol_bulk_nd / self.n_site

    @property
    def vac_bulk_vf(self) -> float:
        """Vacancy bulk volume fraction: fraction of lattice sites unoccupied in bulk."""
        return max(0.0, 1.0 - np.sum(self.ion_bulk_vf) - np.sum(self.sol_bulk_vf))

    @property
    def min_eps(self) -> float:
        """Optical (high-frequency) permittivity, weighted by solvent number density."""
        if self._min_eps_override is not None:
            return self._min_eps_override
        nd = self.sol_bulk_nd
        if len(nd) == 0:
            raise ValueError(
                "min_eps must be provided explicitly when no solvents are present."
            )
        return np.sum(nd * self.get_properties(Solvent, "min_eps")) / np.sum(nd)

    # ── lattice statistical mechanics ─────────────────────────────────────────

    def boltzmann_weights(
        self, phi: np.ndarray, efield: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Single-species statistical weights for the lattice partition function.

          ions:     w_i = exp(-q_i · φ)          (Boltzmann factor)
          solvents: w_s = sinh(p_s·E)/(p_s·E)    (Langevin orientational average)

        φ and E are dimensionless (eφ/kT and e·|E|·Å/kT respectively).

        Returns
        -------
        ion_weights, sol_weights : np.ndarray
            Shape (n_species, ...) matching the shape of phi / E.
        """
        ion_weights = np.exp(-self.ion_q[:, np.newaxis] * phi)
        sol_weights = L.sinh_x_over_x(self.sol_dipole[:, np.newaxis] * efield)
        return ion_weights, sol_weights

    def partition_function(self, phi: np.ndarray, efield: np.ndarray) -> np.ndarray:
        """
        Single-site grand partition function:

            Z = Σ_i f_b,i · w_i  +  Σ_s f_b,s · w_s  +  f_vac

        Z = 1 in the bulk; deviations from 1 encode local crowding and alignment.
        """
        ion_w, sol_w = self.boltzmann_weights(phi, efield)
        return (
            np.sum(self.ion_bulk_vf[:, np.newaxis] * ion_w, axis=0)
            + np.sum(self.sol_bulk_vf[:, np.newaxis] * sol_w, axis=0)
            + self.vac_bulk_vf
        )

    def number_densities(
        self, phi: np.ndarray, efield: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Local number densities from the lattice Boltzmann distribution:

            n_i = n_b,i · w_i / Z    (ions and solvents)

        Returns
        -------
        n_ion, n_sol : np.ndarray
            Shape (n_species, ...) in Å⁻³.
        """
        ion_w, sol_w = self.boltzmann_weights(phi, efield)
        Z = self.partition_function(phi, efield)
        return (
            self.ion_bulk_nd[:, np.newaxis] * ion_w / Z,
            self.sol_bulk_nd[:, np.newaxis] * sol_w / Z,
        )

    def polarization_density(self, phi: np.ndarray, efield: np.ndarray) -> np.ndarray:
        """
        Polarization densities of Langevin dipoles, given by
        Σ_s p_s² · n_s · L(p_s·E)/(p_s·E)
        """
        _, sol_nd = self.number_densities(phi, efield)
        return np.sum(
            self.sol_dipole[:, np.newaxis] ** 2
            * sol_nd
            * L.langevin_x_over_x(self.sol_dipole[:, np.newaxis] * efield),
            axis=0,
        )

    def __str__(self):
        return "\n".join(str(s) for s in self.species)

    def __repr__(self):
        return f"LatticeElectrolyte(species={self.species})"
