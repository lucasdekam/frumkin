"""
Class for storing results
"""

from typing import NamedTuple
import numpy as np


class VoltammetryResult(NamedTuple):
    """
    Class containing results from a Voltammetry experiment
    """

    potential: np.ndarray
    electric_field: np.ndarray
    surface_charge: np.ndarray
    capacitance: np.ndarray
