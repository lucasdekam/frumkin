"""
Class for storing results
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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
        self, fig: Optional[Figure] = None, legend: Optional[int] = None, **kwargs
    ) -> Figure:
        """
        Plot the surface charge and capacitance as a function of potential.

        Parameters
        ----------
        fig : Optional[Figure], optional
            Matplotlib figure to plot on. If None, a new figure will be created.
        legend : Optional[int], optional
            Whether to include a legend in the plot.
        kwargs : dict
            Additional keyword arguments to pass to the plot function.

        Returns
        -------
        Figure
            Matplotlib figure containing the plots.
        """
        if fig is None:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        else:
            axes = np.array(fig.axes)

        y_vals = [self.surface_charge, self.capacitance]
        ylabels = [r"$\sigma$ ($\mu$C/cm$^2$)", r"$C$ ($\mu$F/cm$^2$)"]

        for i, ax in enumerate(axes.reshape(-1)):
            ax.plot(self.potential, y_vals[i], **kwargs)
            ax.set_xlabel("Potential (V)")
            ax.set_ylabel(ylabels[i])
            ax.set_xlim([self.potential.min(), self.potential.max()])

        if legend is not None:
            axes.reshape(-1)[legend].legend()

        fig.tight_layout()
        return fig


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

    def plot(
        self,
        fig: Optional[Figure] = None,
        x_max: float = 20,
        legend: Optional[int] = None,
        **kwargs,
    ) -> Figure:
        """
        Plot the profiles as a function of x.

        Parameters
        ----------
        fig : Optional[Figure], optional
            Matplotlib figure to plot on. If None, a new figure will be created.
        x_max : float, optional
            Maximum x value to plot.
        legend : Optional[int], optional
            Whether to include a legend in the plot.
        kwargs : dict
            Additional keyword arguments to pass to the plot function.

        Returns
        -------
        Figure
            Matplotlib figure containing the plots.
        """
        quantities = {
            r"$\phi$ (V)": self.potential,
            r"$E$ (V/$\AA$)": self.electric_field,
            r"$\varepsilon$": self.permittivity,
        }
        quantities.update(
            {f"[{key}] (M)": value for key, value in self.concentrations.items()}
        )

        nrows = len(quantities) // 2 + len(quantities) % 2

        if fig is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(8, 8))
        else:
            axes = np.array(fig.axes)

        for i, ax in enumerate(axes.reshape(-1)):
            values = list(quantities.values())
            keys = list(quantities.keys())
            ax.plot(self.x, values[i], **kwargs)

            ax.set_xlabel(r"x ($\AA$)")
            ax.set_ylabel(keys[i])
            ax.set_xlim([0, x_max])

        if "label" in kwargs and legend:
            axes.reshape(-1)[0].legend()

        fig.tight_layout()
        return fig
