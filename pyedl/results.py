"""
Class for storing results
"""

from typing import NamedTuple, Dict
import numpy as np


class VoltammetryResult(NamedTuple):
    """
    Class containing results from a Voltammetry experiment
    """

    potential: np.ndarray
    electric_field: np.ndarray
    surface_charge: np.ndarray
    capacitance: np.ndarray


class SinglePointResult(NamedTuple):
    """
    Class containing profiles from a single point experiment
    """

    x: np.ndarray
    potential: np.ndarray
    electric_field: np.ndarray
    permittivity: np.ndarray
    concentrations: Dict
