"""
Plotting tools for double-layer models
"""

import matplotlib.pyplot as plt
import spatial_profiles as prf

def plot_solution(sol: prf.SpatialProfilesSolution, xmin: float, xmax: float):
    """
    Plot spatial profiles
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,4), sharex=True)
    ax[0].plot(sol.x, sol.phi, label=sol.name, color='black')
    ax[0].set_xlabel(r'$x$ [nm]')
    ax[0].set_ylabel(r'$\phi$ [V vs. PZC]')
    ax[0].set_xlim([xmin, xmax])
    ax[0].legend()

    ax[1].plot(sol.x, sol.c_cat, color='tab:blue', label='Cations')
    ax[1].plot(sol.x, sol.c_an, color='tab:red', label='Anions')
    ax[1].set_yscale('log')
    ax[1].legend(loc='upper right')
    ax[1].set_ylabel('c [M]')
    ax[1].set_xlabel(r'$x$ [nm]')

    ax[2].plot(sol.x, sol.eps)
    ax[2].set_ylim([0, 100])
    ax[2].set_ylabel(r'$\varepsilon_r$')
    ax[2].set_xlabel(r'$x$ [nm]')
    plt.tight_layout()

    return fig, ax