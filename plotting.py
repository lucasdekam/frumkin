"""
Plotting tools for double-layer models
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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
        sol,
        xmin: float, xmax: float,
        logscale=True):
    """
    Plot spatial profiles
    """
    # pylint: disable=invalid-name
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,4), sharex=True)

    ax[0] = plot_phi(ax[0], sol.name, sol.x, sol.phi)
    # ax[0] = plot_phi(ax[0], sol.name, sol.x, sol.efield)
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
        sol1,
        sol2,
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


def plot_potential_sweep(sols: list):
    """
    Plot charge and differential capacitance
    """
    # pylint: disable=invalid-name
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4), sharex=True)

    for sol in sols:
        ax[0].plot(sol.phi, sol.charge * 1e2, label=sol.name)
    ax[0].set_ylabel(r'$\sigma$ [$\mu$C/cm$^2$]')
    ax[0].set_xlabel(r'$\phi$ [V vs. PZC]')
    ax[0].legend()
    ax[0].set_ylim([-50, 50])

    for sol in sols:
        ax[1].plot(sol.phi, sol.cap, label=sol.name)
    ax[1].set_ylabel(r'Capacitance [$\mu$F/cm$^2$]')
    ax[1].set_xlabel(r'$\phi$ [V vs. PZC]')
    ax[1].legend()
    ax[1].set_ylim([0, 150])

    plt.tight_layout()
    return fig, ax

def get_color_gradient(array: np.ndarray):
    """
    Get a light blue to dark blue color gradient in the form of
    an array of RGB tuples
    [(r1, g1, b1), ..., (rn, gn, bn)]
    """
    red = np.linspace(3, 2, len(array))[::-1]/255
    gre = np.linspace(57, 242, len(array))[::-1]/255
    blu = np.linspace(143, 250, len(array))[::-1]/255

    return [(red[i], gre[i], blu[i]) for i in range(len(array))]

def plot_current(current_sweeps: np.ndarray,
                 potential_range: np.ndarray,
                 parameter_range: np.ndarray,
                 parameter_symbol: str,
                 parameter_scaling: float,
                 parameter_unit: str,
                 ylabel: str=r'$j/|j_\mathrm{max}|$',
                 xlabel: str=r'$\phi_0$ [V] vs. RHE'):
    """
    Plot kinetics
    currents: shape (len(parameter_range), len(potential_range))
    """
    fig, ax = plt.subplots(figsize=(4,3))
    for i, current in enumerate(current_sweeps):
        ax.plot(potential_range, current,
                color=get_color_gradient(parameter_range)[i],
                label=rf'{parameter_symbol}=' + \
                rf'{parameter_range[i]*parameter_scaling:.0f} {parameter_unit}')
    ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    plt.tight_layout()
    return fig, ax
