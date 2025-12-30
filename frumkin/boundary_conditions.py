"""
Boundary conditions for Poisson-Boltzmann models
"""

import numpy as np
from scipy import constants
from .tools.langevin import langevin_x


def waterlayer(
    ya: np.ndarray,
    yb: np.ndarray,
    y0: float,
    eps_ohp: float,
    **kwargs,
):
    eps_1 = kwargs.get("eps_1", 10)
    eps_2 = kwargs.get("eps_2", 3.25)
    eps_3 = kwargs.get("eps_3", 78)  # 35)
    d_1 = kwargs.get("d_1", 1)
    d_2 = kwargs.get("d_2", 1)  # 1.1
    d_3 = kwargs.get("d_3", 2)  # 3

    n_sites = kwargs.get("n_sites", 0.139)
    water_coverage = kwargs.get("water_coverage", 0.55)
    dipole_debye = kwargs.get("dipole_debye", 0.73)
    dip = dipole_debye * 3.335e-30 / constants.elementary_charge / constants.angstrom

    temperature = kwargs.get("temperature", 298)
    kbt = constants.Boltzmann * temperature
    kappa = (
        constants.elementary_charge**2 / constants.epsilon_0 / constants.angstrom / kbt
    )
    delta_chemi = kwargs.get("delta_chemi", 0)

    dy_water = (
        n_sites
        * water_coverage
        * dip
        * kappa
        * langevin_x(dip * ya[1] * eps_ohp / eps_2 + delta_chemi).item()
        / eps_2
    )

    # y_onset = constants.elementary_charge * kwargs.get("E_onset", 0.5) / kbt
    # oxide_charge = kwargs.get("oxide_charge", 0.12)
    # oxide_coverage = np.exp(-(y_onset - y0)) / (1 + np.exp(-(y_onset - y0)))
    # dy_adsorbate = kappa * d_1 / eps_1 * n_sites * oxide_coverage * oxide_charge

    return np.array(
        [
            ya[0].item()
            - y0
            - ya[1].item() * eps_ohp * (d_1 / eps_1 + d_2 / eps_2 + d_3 / eps_3)
            + dy_water,  # - dy_adsorbate,
            yb[0],
        ]
    )


def semi_infinite(
    ya: np.ndarray,
    yb: np.ndarray,
    y0: float,
    ohp: float,
    eps_ratio: float,
):
    """
    Semi-infinite system: electrode on the left side (x=0),
    bulk on the right side (x=infinity, in practice x=large enough)

    Parameters
    ----------
    ya: [y, dy/dx] at the left boundary
    yb: [y, dy/dx] at the right boundary
    y0: dimensionless applied potential (beta e phi) at the electrode
    ohp: thickness of the Stern layer, i.e., position of the outer Helmholtz plane
    eps_ratio: ratio between the diffuse and Stern layer permittivities
    """
    return np.array(
        [
            ya[0] - y0 - ya[1] * ohp * eps_ratio,
            yb[0],
        ]
    )


def symmetric(
    ya: np.ndarray,
    yb: np.ndarray,
    y0: float,
    ohp: float,
    eps_ratio: float,
):
    """
    Symmetric system: equally charged electrodes on the left and right side

    Parameters
    ----------
    ya: [y, dy/dx] at the left boundary
    yb: [y, dy/dx] at the right boundary
    y0: dimensionless applied potential (beta e phi) at the electrode
    ohp: thickness of the Stern layer, i.e., position of the outer Helmholtz plane
    eps_ratio: ratio between the diffuse and Stern layer permittivities
    """
    return np.array(
        [
            ya[0] - y0 - ya[1] * ohp * eps_ratio,
            y0 - yb[0] - yb[1] * ohp * eps_ratio,
        ]
    )


def antisymmetric(
    ya: np.ndarray,
    yb: np.ndarray,
    y0: float,
    ohp: float,
    eps_ratio: float,
):
    """
    Antisymmetric system: two electrodes with equal but opposite charge, one on the
    left and one on the right

    Parameters
    ----------
    ya: [y, dy/dx] at the left boundary
    yb: [y, dy/dx] at the right boundary
    y0: dimensionless applied potential (beta e phi) at the electrode
    ohp: thickness of the Stern layer, i.e., position of the outer Helmholtz plane
    eps_ratio: ratio between the diffuse and Stern layer permittivities
    """
    return np.array(
        [
            ya[0] - y0 - ya[1] * ohp * eps_ratio,
            -y0 - yb[0] - yb[1] * ohp * eps_ratio,
        ]
    )
