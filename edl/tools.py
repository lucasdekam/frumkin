"""
Boundary value problem tools
"""
import numpy as np


def create_mesh_m(xmax_nm, n_points):
    """
    Get a logarithmically spaced x-axis, fine mesh close to electrode
    """
    max_exponent = np.log10(xmax_nm)
    x_nm = np.logspace(-6, max_exponent, n_points) - 1e-6
    return x_nm * 1e-9
