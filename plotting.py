"""
Plotting tools for double-layer models
"""

from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import spatial_profiles as prf


def plot_phi(ax: Axes, name: str, x: np.ndarray, phi: np.ndarray, fmt='-', color='black'):
    """
    Plot potential in an axis object
    """
    # pylint: disable=invalid-name
    ax.plot(x, phi, fmt, color=color, label=name)
    ax.set_xlabel(r'$x$ [nm]')
    ax.set_ylabel(r'$\phi$ [V vs. PZC]')
    ax.legend()
    return ax

def plot_concentrations(ax: Axes, x: np.ndarray, c_dict: dict, fmt: str='-', logscale: bool=False):
    """
    Plot concentration profiles
    """
    # pylint: disable=invalid-name
    lines = []
    for name, profile in c_dict.items():
        if name == 'Solvent':
            if np.sum(np.abs(profile)) != 0:
                if logscale: # if logscale, no twin axis necessary
                    line = ax.plot(x, profile, fmt, color='gray', label=name)
                else: # if not logscale, make a twin axis
                    sol_ax = ax.twinx()
                    line = sol_ax.plot(x, profile, fmt, color='gray', label=name)
                    sol_ax.set_ylim([0, np.max(sum(list(c_dict.values())))*1.1])
                lines.append(line)
        else:
            line = ax.plot(x, profile, fmt, label=name)
            lines.append(line)

    all_lines = lines[0]
    for l in lines[1:]:
        all_lines += l

    labels=[l.get_label() for l in all_lines]
    ax.legend(all_lines, labels)
    ax.set_ylabel('c [M]')
    ax.set_xlabel(r'$x$ [nm]')
    if logscale:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    return ax, all_lines

def plot_permittivity(ax: Axes, x: np.ndarray, eps: np.ndarray, fmt='-', color='tab:blue'):
    """
    Plot permittivity profile
    """
    # pylint: disable=invalid-name
    ax.plot(x, eps, fmt, color=color)
    ax.set_ylim([0, 100])
    ax.set_ylabel(r'$\varepsilon_r$')
    ax.set_xlabel(r'$x$ [nm]')
    return ax

def plot_solution(
        sol: prf.SpatialProfilesSolution,
        xmin: float, xmax: float,
        logscale=True):
    """
    Plot spatial profiles
    """
    # pylint: disable=invalid-name
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,4), sharex=True)

    ax[0] = plot_phi(ax[0], sol.name, sol.x, sol.phi)
    ax[0].set_xlim([xmin, xmax])
    ax[1], _ = plot_concentrations(
        ax[1],
        sol.x,
        sol.c_dict,
        logscale=logscale)
    ax[2] = plot_permittivity(ax[2], sol.x, sol.eps)
    plt.tight_layout()
    return fig, ax


def plot_sol_comparison(
        sol1: prf.SpatialProfilesSolution,
        sol2: prf.SpatialProfilesSolution,
        xmin: float, xmax: float,
        logscale=True):
    """
    Compare two solutions
    """
    # pylint: disable=invalid-name
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,4), sharex=True)

    ax[0] = plot_phi(ax[0], sol1.name, sol1.x, sol1.phi, fmt='-', color='black')
    ax[0] = plot_phi(ax[0], sol2.name, sol2.x, sol2.phi, fmt='--', color='gray')
    ax[0].set_xlim([xmin, xmax])

    ax[1], lines1 = plot_concentrations(
        ax[1],
        sol1.x,
        sol1.c_dict,
        fmt='-',
        logscale=logscale)
    ax[1], lines2 = plot_concentrations(
        ax[1],
        sol2.x,
        sol2.c_dict,
        fmt='--',
        logscale=logscale)

    all_lines = lines1 + lines2
    labels=[l.get_label() for l in all_lines]
    ax[1].legend(all_lines, labels)

    ax[2] = plot_permittivity(ax[2], sol1.x, sol1.eps, fmt='-', color='black')
    ax[2] = plot_permittivity(ax[2], sol2.x, sol2.eps, fmt='--', color='gray')

    plt.tight_layout()
    return fig, ax