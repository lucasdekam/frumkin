"""
Plotting tools for double-layer models
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def plot_phi(ax: Axes, name: str, x: np.ndarray, phi: np.ndarray, color='black', suppress_names=False):
    """
    Plot potential in an axis object
    """
    # pylint: disable=invalid-name
    if suppress_names:
        ax.plot(x, phi, color=color)
    else:
        ax.plot(x, phi, color=color, label=name)
        ax.legend()
    ax.set_xlabel(r'$x$ [nm]')
    ax.set_ylabel(r'$\phi$ [V vs. PZC]')
    return ax

def plot_efield(ax: Axes, x: np.ndarray, efield: np.ndarray, color='black'):
    """
    Plot potential in an axis object
    """
    # pylint: disable=invalid-name
    ax.plot(x, efield * 1e-9, color=color)
    ax.set_xlabel(r'$x$ [nm]')
    ax.set_ylabel(r'$E$ [V/nm]')
    return ax

def plot_concentrations(ax: Axes,
                        spec : str,
                        x: np.ndarray,
                        c: np.ndarray,
                        fmt: str='-',
                        color='tab:blue',
                        logscale: bool=False):
    """
    Plot concentration profiles
    """
    # pylint: disable=invalid-name
    ax.plot(x, c, fmt, color=color)
    ax.set_ylabel(spec + ' [M]')
    ax.set_xlabel(r'$x$ [nm]')
    if logscale:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    return ax

def plot_permittivity(ax: Axes, x: np.ndarray, eps: np.ndarray, color='tab:red'):
    """
    Plot permittivity profile
    """
    # pylint: disable=invalid-name
    ax.plot(x, eps, color=color)
    ax.set_ylim([0, 100])
    ax.set_ylabel(r'$\varepsilon_r$')
    ax.set_xlabel(r'$x$ [nm]')
    return ax

def plot_solutions(
        sol_list: list[pd.DataFrame],
        xmin: float, xmax: float,
        suppress_names=False):
    """
    Plot spatial profiles
    """
    # pylint: disable=invalid-name
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9,5), sharex=True)

    colors = get_color_gradient(len(sol_list))

    for i, sol in enumerate(sol_list):
        ax[0,0] = plot_phi(ax[0,0],
                           sol.index.name,
                           sol['x'],
                           sol['phi'],
                           color=colors[i],
                           suppress_names=suppress_names)
        ax[0,1] = plot_efield(ax[0,1],
                              sol['x'],
                              sol['efield'],
                              color=colors[i])
        ax[0,2] = plot_permittivity(ax[0,2], sol.x, sol.eps, color=colors[i])
        ax[1,0] = plot_concentrations(ax[1,0],
                                      'Cations',
                                      sol['x'],
                                      sol['cations'],
                                      color=colors[i])
        if 'hydroxide' in sol.columns:
            ax[1,1] = plot_concentrations(ax[1,1],
                                        'Anions',
                                        sol['x'],
                                        sol['hydroxide'] + sol['anions'],
                                        color=colors[i])
        else:
            ax[1,1] = plot_concentrations(ax[1,1],
                                        'Anions',
                                        sol['x'],
                                        sol['anions'],
                                        color=colors[i])
        if 'solvent' in sol.columns:
            ax[1,2] = plot_concentrations(ax[1,2],
                                        'Solvent',
                                        sol['x'],
                                        sol['solvent'],
                                        color=colors[i])
    ax[0,0].set_xlim([xmin, xmax])

    plt.tight_layout()
    return fig, ax


def plot_potential_sweep(sols: list[pd.DataFrame]):
    """
    Plot charge and differential capacitance
    """
    # pylint: disable=invalid-name
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4), sharex=True)

    for sol in sols:
        ax[0].plot(sol['phi0'], sol['charge'] * 1e2, label=sol.index.name)
    ax[0].set_ylabel(r'$\sigma$ [$\mu$C/cm$^2$]')
    ax[0].set_xlabel(r'$\phi$ [V vs. PZC]')
    ax[0].legend()
    ax[0].set_ylim([-50, 50])

    for sol in sols:
        ax[1].plot(sol['phi0'], sol['capacity'] * 1e2, label=sol.index.name)
    ax[1].set_ylabel(r'Capacitance [$\mu$F/cm$^2$]')
    ax[1].set_xlabel(r'$\phi$ [V vs. PZC]')
    ax[1].legend()
    ax[1].set_ylim([0, 150])

    plt.tight_layout()
    return fig, ax

def get_color_gradient(size: int, color='blue'):
    """
    Get a light blue to dark blue color gradient in the form of
    an array of RGB tuples
    [(r1, g1, b1), ..., (rn, gn, bn)]
    """
    if color=='blue':
        red = np.linspace(3, 2, size)[::-1]/255
        gre = np.linspace(57, 242, size)[::-1]/255
        blu = np.linspace(143, 250, size)[::-1]/255
    elif color=='green':
        red = np.linspace(21, 97, size)[::-1]/255
        gre = np.linspace(99, 214, size)[::-1]/255
        blu = np.linspace(9, 79, size)[::-1]/255
    else:
        return None

    return [(red[i], gre[i], blu[i]) for i in range(size)]

def plot_current(current_sweeps: np.ndarray,
                 potential_range: np.ndarray,
                 parameter_range: np.ndarray,
                 parameter_symbol: str,
                 parameter_scaling: float=1,
                 parameter_unit: str='',
                 ylabel: str=r'$j/|j_\mathrm{max}|$',
                 xlabel: str=r'$\phi_0$ [V] vs. RHE'):
    """
    Plot kinetics
    currents: shape (len(parameter_range), len(potential_range))
    """
    fig, ax = plt.subplots(figsize=(4,3))
    for i, current in enumerate(current_sweeps):
        ax.plot(potential_range, current,
                color=get_color_gradient(len(parameter_range))[i],
                label=rf'{parameter_symbol}=' + \
                rf'{parameter_range[i]*parameter_scaling:.0f} {parameter_unit}')
    ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    plt.tight_layout()
    return fig, ax
