"""
Plotting tools for double-layer models
"""

import numpy as np
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

    ln1 = ax[1].plot(sol.x, sol.c_cat, color='tab:blue', label='Cations')
    ln2 = ax[1].plot(sol.x, sol.c_an, color='tab:red', label='Anions')

    if plot_water and not logscale:
        waterax = ax[1].twinx()
        ln3=waterax.plot(sol.x, sol.c_sol, '-', color='gray', label='Solvent')
        waterax.set_ylim([0, np.max(sol.c_an+sol.c_cat+sol.c_sol)+0.1])
        waterax.set_ylabel(r'$c_\mathrm{sol}$ [M]')
        # ln3=ax[1].plot(sol.x, sol.c_sol, '-', color='gray', label='Solvent sites')
        # ax[1].plot(sol.x, sol.c_an+sol.c_cat+sol.c_sol, '--', color='tab:green')
        lns = ln1+ln2+ln3
        labs = [l.get_label() for l in lns]
        ax[1].legend(lns, labs)
    elif plot_water and logscale:
        ax[1].plot(sol.x, sol.c_sol, '-', color='gray', label='Solvent')
        ax[1].legend()
    else:
        ax[1].legend()

    if logscale:
        ax[1].set_yscale('log')
    else:
        ax[1].set_yscale('linear')

    ax[1].set_ylabel('c [M]')
    ax[1].set_xlabel(r'$x$ [nm]')

    ax[2].plot(sol.x, sol.eps)
    ax[2].set_ylim([0, 100])
    ax[2].set_ylabel(r'$\varepsilon_r$')
    ax[2].set_xlabel(r'$x$ [nm]')
    plt.tight_layout()

    return fig, ax