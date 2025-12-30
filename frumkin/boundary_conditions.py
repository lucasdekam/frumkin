"""
Boundary conditions for Poisson-Boltzmann models
"""

import numpy as np
from scipy import constants
from .tools.langevin import langevin_x


import numpy as np
from scipy import constants


def get_stern_components(ya, eps_ohp, params):
    """
    Shared logic to calculate the field-dependent terms in the Stern layer.
    ya[1] is the dimensionless potential gradient at the OHP.
    """
    # 1. Extract physical constants/parameters
    d_vals = [params.get("d_1", 1.0), params.get("d_2", 1.1), params.get("d_3", 1.0)]
    eps_vals = [
        params.get("eps_1", 10.0),
        params.get("eps_2", 3.25),
        params.get("eps_3", 78.0),
    ]

    n_sites = params.get("n_sites", 0.139)
    water_coverage = params.get("water_coverage", 0.55)
    dipole_debye = params.get("dipole_debye", 0.75)
    temp = params.get("temperature", 298)
    delta_chemi = params.get("delta_chemi", 0)

    # Conversion factors for dipole potential drop
    # dip is in units of [e * Angstrom]
    dip = dipole_debye * 3.335e-30 / constants.elementary_charge / constants.angstrom
    kbt = constants.Boltzmann * temp
    kappa = constants.elementary_charge**2 / (
        constants.epsilon_0 * constants.angstrom * kbt
    )

    # 2. Calculate Potential Drop due to Water Dipoles (dy_water)
    # Applied at layer 2 (the water layer)
    # Langevin argument depends on the local field
    langevin_arg = dip * ya[1].item() * eps_ohp / eps_vals[1] + delta_chemi

    dy_dipole = (
        n_sites * water_coverage * dip * kappa * langevin_x(langevin_arg)
    ) / eps_vals[1]

    # 3. Calculate Potential Drop due to Free Charge (ya[1] * eps_ohp)
    # Delta_phi_i = E_i * d_i = (sigma_free / epsilon_i) * d_i
    # In your dimensionless code: ya[1] * eps_ohp / eps_i * d_i
    phi_drops_charge = [
        -(ya[1].item() * eps_ohp / e) * d for e, d in zip(eps_vals, d_vals)
    ]

    return phi_drops_charge, dy_dipole, d_vals


def waterlayer(ya, yb, y0, eps_ohp, **kwargs):
    """
    Boundary condition for the BVP solver.
    Ensures: phi_M - sum(drops) = phi_OHP
    """
    phi_drops, dy_dipole, _ = get_stern_components(ya, eps_ohp, kwargs)

    # Total potential drop across all Stern slabs
    total_drop = (
        sum(phi_drops) + dy_dipole
    )  # Subtracting because dipoles usually oppose the field

    return np.array(
        [
            ya[0].item() - (y0 - total_drop.item()),
            yb[0].item(),  # Potential at infinity (bulk) is 0
        ]
    )


def calculate_stern_profile(ya, y0, eps_ohp, **kwargs):
    """
    Calculates the (x, y) coordinates of the potential through the Stern layer.
    Returns: x_coords (distance from electrode), y_coords (potential)
    """
    phi_drops, dy_dipole, d_vals = get_stern_components(ya, eps_ohp, kwargs)

    # x=0 is the electrode surface (Metal)
    x = [0]
    y = [y0]

    # Layer 1
    x.append(x[-1] + d_vals[0])
    y.append(y[-1] - phi_drops[0].item())

    # Layer 2 (Water layer - includes the dipole jump at the end of the layer)
    x.append(x[-1] + d_vals[1])
    # The dipole contribution is often modeled as a sheet at the interface
    # or distributed; here we apply it at the end of layer 2.
    y.append(y[-1] - phi_drops[1].item() + dy_dipole.item())

    # Layer 3
    x.append(x[-1] + d_vals[2])
    y.append(y[-1] - phi_drops[2].item())

    return np.array(x), np.array(y)


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
