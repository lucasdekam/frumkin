"""
mesh.py

Meshing for solving Poisson-Boltzmann equations in 1D
"""

import numpy as np


def create_mesh(xmax=1000, n_points=1000):
    """
    Generate a logarithmically spaced mesh for the x-axis.

    Parameters
    ----------
    xmax : int or float, optional
        The maximum value of the x-axis. Default is 1000.
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
