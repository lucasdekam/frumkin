"""
Stern layer boundary conditions for Poisson-Boltzmann models.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy import constants
from .tools.langevin import langevin_x


class SternModel(ABC):
    """
    Abstract Stern-layer model.

    Subclasses implement the relationship between the diffuse-side field at
    the OHP and (a) the total potential drop across the Stern region, and
    (b) the spatial profile of potential, field, and permittivity through
    that region.
    """

    @property
    @abstractmethod
    def ohp(self) -> float:
        """Total Stern thickness (distance from electrode to OHP)."""

    @abstractmethod
    def drop(self, yp_ohp: float, eps_diffuse: float) -> float:
        """
        Potential drop phi_OHP - phi_metal given the dimensionless diffuse-side
        field dy/dx at the OHP and the diffuse-side relative permittivity.
        """

    @abstractmethod
    def profile(self, y_ohp: np.ndarray, y_metal: float, eps_diffuse: float):
        """
        Return (x, y, eps) sampled from the electrode (x=0) to the OHP
        (x=self.ohp), with y_metal the dimensionless potential at the electrode.
        y_ohp has shape (2,) and y has shape (2,npoints): y[0] is the potential,
        y[1] is its derivative.
        """


class SimpleStern(SternModel):
    """
    Single-slab Stern layer with constant permittivity. If `eps_stern` is None,
    the Stern permittivity is taken to equal the diffuse-layer permittivity
    (i.e. no permittivity discontinuity at the OHP).
    """

    def __init__(self, ohp: float, eps_stern: Optional[float] = None):
        self._ohp = ohp
        self.eps_stern = eps_stern

    @property
    def ohp(self) -> float:
        return self._ohp

    def _eps_ratio(self, eps_diffuse: float) -> float:
        return 1.0 if self.eps_stern is None else eps_diffuse / self.eps_stern

    def drop(self, yp_ohp, eps_diffuse):
        # No free charges => constant field across the slab
        # => drop = y_ohp - y_metal = -y'_stern * d
        # at OHP: eps_stern * y'_stern = eps_diffuse * y'_diffuse
        # (continuity of D)
        # => drop = y'_diffuse * eps_diffuse / eps_stern * d
        return yp_ohp * self._ohp * self._eps_ratio(eps_diffuse)

    def profile(self, y_ohp, y_metal, eps_diffuse):
        n = 10
        x = np.linspace(0, self._ohp, n)
        eps_val = self.eps_stern if self.eps_stern is not None else eps_diffuse
        eps = np.full(n, eps_val)
        # Linear from y_metal at x=0 down to y_metal - drop at x=ohp.
        y = np.linspace(y_metal, y_ohp[0], n)
        yp = np.full(n, y_ohp[1] * self._eps_ratio(eps_diffuse))
        return x, np.vstack([y, yp]), eps


class WaterLayer(SternModel):
    """
    Three-slab Stern model with a Langevin water-dipole potential drop
    located at the outer edge of the middle (water) layer.

    Reference: https://pubs.acs.org/doi/10.1021/jacsau.2c00650

    Parameters
    ----------
    d : sequence of 3 floats
        Slab thicknesses (Angstrom) for layers 1, 2 (water), 3.
    eps : sequence of 3 floats
        Relative permittivities of the three slabs.
    n_sites, water_coverage : float
        Surface density of dipole sites and fraction occupied by oriented water.
    dipole_debye : float
        Magnitude of the water dipole moment in Debye.
    temperature : float
        Temperature in K.
    delta_chemi : float
        Chemisorption-induced offset added to the Langevin argument.
    """

    def __init__(
        self,
        d=(1.0, 1.1, 1.0),
        eps=(10.0, 3.25, 78.0),
        n_sites: float = 0.139,
        water_coverage: float = 0.55,
        dipole_debye: float = 0.75,
        temperature: float = 298.0,
        delta_chemi: float = 0.0,
    ):
        self.d = np.asarray(d, dtype=float)
        self.eps = np.asarray(eps, dtype=float)
        if self.d.shape != (3,) or self.eps.shape != (3,):
            raise ValueError("WaterLayerStern expects exactly 3 slabs.")
        self.n_sites = n_sites
        self.water_coverage = water_coverage
        self.dipole_debye = dipole_debye
        self.temperature = temperature
        self.delta_chemi = delta_chemi

    @property
    def ohp(self) -> float:
        return float(self.d.sum())

    def _dip_e_angstrom(self):
        """Dipole moment in units of [e * Angstrom]."""
        return (
            self.dipole_debye
            * 3.335e-30
            / constants.elementary_charge
            / constants.angstrom
        )

    def _kappa(self):
        """e^2 / (eps_0 * Angstrom * kT)."""
        kbt = constants.Boltzmann * self.temperature
        return constants.elementary_charge**2 / (
            constants.epsilon_0 * constants.angstrom * kbt
        )

    def _components(self, yp_ohp: float, eps_diffuse: float):
        """
        Returns
        -------
        drops : (3,) array
            Free-charge potential drops across each slab; positive when phi
            decreases moving from electrode to bulk.
        dy_dipole : float
            Potential jump from oriented water dipoles at the outer edge of
            the water layer.
        """
        dip = self._dip_e_angstrom()
        kappa = self._kappa()

        # Local field inside the water layer (continuity of D):
        # E_water = E_diffuse * eps_diffuse / eps_water
        langevin_arg = dip * yp_ohp * eps_diffuse / self.eps[1] + self.delta_chemi

        dy_dipole = (
            self.n_sites * self.water_coverage * dip * kappa * langevin_x(langevin_arg)
        ) / self.eps[1]

        drops = np.array(
            [yp_ohp * eps_diffuse / e * d for e, d in zip(self.eps, self.d)]
        )
        return drops, dy_dipole

    def drop(self, yp_ohp, eps_diffuse):
        yp = yp_ohp.item() if hasattr(yp_ohp, "item") else float(yp_ohp)
        drops, dy_dipole = self._components(yp, eps_diffuse)
        return float(drops.sum()) + float(dy_dipole)

    def profile(self, y_ohp, y_metal, eps_diffuse):
        yp_ohp = y_ohp[1]
        yp_ohp = yp_ohp.item() if hasattr(yp_ohp, "item") else float(yp_ohp)
        drops, dy_dipole = self._components(yp_ohp, eps_diffuse)

        n_per_slab = 5
        xs, ys, yps, es = [], [], [], []

        x_start = 0.0
        y_start = y_metal
        for i, (d_i, eps_i, drop_i) in enumerate(zip(self.d, self.eps, drops)):
            x_slab = np.linspace(x_start, x_start + d_i, n_per_slab)
            y_slab = np.linspace(y_start, y_start - drop_i, n_per_slab)
            yp_slab = np.full(n_per_slab, yp_ohp * eps_diffuse / eps_i)
            eps_slab = np.full(n_per_slab, eps_i)

            xs.append(x_slab)
            ys.append(y_slab)
            yps.append(yp_slab)
            es.append(eps_slab)

            x_start += d_i
            y_start -= drop_i

            # Dipole jump as a sheet at the outer edge of the water layer.
            if i == 1:
                xs.append(np.array([x_start]))
                y_after = y_start + dy_dipole
                ys.append(np.array([y_after]))
                yps.append(np.array([yp_ohp * eps_diffuse / self.eps[2]]))
                es.append(np.array([self.eps[1]]))
                y_start = y_after

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        yp = np.concatenate(yps)
        eps = np.concatenate(es)
        return x, np.vstack([y, yp]), eps
