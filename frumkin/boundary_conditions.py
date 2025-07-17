"""
Boundary conditions for Poisson-Boltzmann models
"""

import numpy as np


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
