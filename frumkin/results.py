"""
Class for storing results
"""

from dataclasses import dataclass
from typing import Dict, Optional, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


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
    """

    potential: np.ndarray
    surface_charge: np.ndarray
    capacitance: np.ndarray

    def plot(
        self,
        ax: Axes,
        y: Literal["capacitance", "surface charge"] = "capacitance",
        **kwargs,
    ) -> None:
        """
        Plot the surface charge and capacitance as a function of potential.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes to plot on. If None, a new figure will be created.
        kwargs : dict
            Additional keyword arguments to pass to the plot function.
        """
        ylabels = {
            "surface charge": r"$\sigma$ ($\mu$C/cm$^2$)",
            "capacitance": r"$C$ ($\mu$F/cm$^2$)",
        }

        if y == "capacitance":
            ax.plot(self.potential, self.capacitance, **kwargs)
        elif y == "surface charge":
            ax.plot(self.potential, self.surface_charge, **kwargs)

        ax.set_xlabel("Potential (V)")
        ax.set_ylabel(ylabels[y])
        ax.set_xlim([self.potential.min(), self.potential.max()])


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
