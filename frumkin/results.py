"""
Class for storing results
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class VoltammetryResult:
    """
    Class containing results from a Voltammetry experiment.

    Attributes
    ----------
    potential : np.ndarray
        Potential values in V at which the results are calculated.
    surface_charge : np.ndarray
        Surface charge in uC/cm^2.
    capacitance : np.ndarray
        Capacitance in uF/cm^2.
    electric_field : np.ndarray
    """

    potential: np.ndarray
    surface_charge: np.ndarray
    capacitance: np.ndarray
    electric_field: np.ndarray


@dataclass
class SinglePointResult:
    """
    Class containing profiles from a single point experiment.

    Attributes
    ----------
    x : np.ndarray
        x values in Angstroms.
    potential : np.ndarray
        Potential values in V.
    electric_field : np.ndarray
        Electric field values in V/Å.
    permittivity : np.ndarray
        Relative permittivity.
    concentrations : Dict
        Dictionary of concentrations.
    """

    x: np.ndarray
    potential: np.ndarray
    electric_field: np.ndarray
    permittivity: np.ndarray
    concentrations: Dict
