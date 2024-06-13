"""
Plotting tools for double-layer models
"""

import numpy as np

GAMMA_LIST = [4, 5, 6, 7]
CONC_LIST = [1e-3, 1e-2, 1e-1, 1]
DEFAULT_GAMMA = 6
DEFAULT_CONC_M = 100e-3


def get_color_gradient(size: int, color="blue"):
    """
    Get a light blue to dark blue color gradient in the form of
    an array of RGB tuples
    [(r1, g1, b1), ..., (rn, gn, bn)]
    """
    if color == "blue":
        red = np.linspace(3, 2, size)[::-1] / 255
        gre = np.linspace(57, 242, size)[::-1] / 255
        blu = np.linspace(143, 250, size)[::-1] / 255
    elif color == "green":
        red = np.linspace(21, 97, size)[::-1] / 255
        gre = np.linspace(99, 214, size)[::-1] / 255
        blu = np.linspace(9, 79, size)[::-1] / 255
    elif color == "red":
        red = np.linspace(59, 252, size)[::-1] / 255
        gre = np.linspace(3, 45, size)[::-1] / 255
        blu = np.linspace(28, 52, size)[::-1] / 255
    elif color == "purple":
        red = np.linspace(56, 230, size)[::-1] / 255
        gre = np.linspace(4, 118, size)[::-1] / 255
        blu = np.linspace(107, 245, size)[::-1] / 255
    elif color == "black":
        red = np.linspace(18, 163, size)[::-1] / 255
        gre = red
        blu = red
    else:
        return None

    return [(red[i], gre[i], blu[i]) for i in range(size)]
