"""
mesh.py

Meshing for solving Poisson-Boltzmann equations in 1D
"""

from typing import Literal
import numpy as np


def create_semi_infinite_mesh(xmax: float = 1000, n_points: int = 1000):
    """
    Generate a logarithmically spaced mesh for the x-axis.

    Parameters
    ----------
    xmax : int or float, optional
        The maximum value of the x-axis (in Angstrom). Default is 1000.
    n_points : int, optional
        The number of points in the mesh. Default is 1000.

    Returns
    -------
    numpy.ndarray
        A logarithmically spaced array of x-axis values, fine mesh close to the electrode.
    """

    max_exponent = np.log10(xmax)
    mesh = np.logspace(-6, max_exponent, n_points) - 1e-6
    return mesh


def create_symmetric_mesh(xmax=100, n_points=1000):
    return np.linspace(0, xmax, n_points)


def get_default_mesh(
    boundary: Literal["semi-infinite", "symmetric", "antisymmetric"],
    xmax,
):
    if boundary == "semi-infinite":
        return create_semi_infinite_mesh(xmax)
    else:
        return create_symmetric_mesh(xmax)
