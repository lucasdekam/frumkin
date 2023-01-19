"""
Plotting tools for double-layer models
"""

import matplotlib.pyplot as plt
import spatial_profiles as prf

def plot_solution(sol: prf.SpatialProfilesSolution, xmin: float, xmax: float, logscale=True, plot_water: bool=False):
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

    if plot_water and not logscale:
        waterax = ax[1].twinx()
        waterax.plot(sol.x, sol.c_sol, '-', color='gray')
        waterax.set_ylim([0, 60])
    elif plot_water and logscale:
        ax[1].plot(sol.x, sol.c_sol, '-', color='gray', label='Solvent')

    if logscale:
        ax[1].set_yscale('log')
    else:
        ax[1].set_yscale('linear')

    ax[1].legend(loc='lower right')
    ax[1].set_ylabel('c [M]')
    ax[1].set_xlabel(r'$x$ [nm]')

    ax[2].plot(sol.x, sol.eps)
    ax[2].set_ylim([0, 100])
    ax[2].set_ylabel(r'$\varepsilon_r$')
    ax[2].set_xlabel(r'$x$ [nm]')
    plt.tight_layout()

    return fig, ax