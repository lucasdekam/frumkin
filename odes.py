"""
ODE tools for double-layer models
"""
import os
import pickle
import numpy as np
from scipy.integrate import solve_bvp

def get_solution(x_axis, ode, boundary_condition, model_name, spec_string, verbose=False, force_recalculation=False):
    """
    Solve the ODE system or load the solution if there is one already
    """
    y = np.zeros((2, x_axis.shape[0]))

    # Make directory for solutions if there is none existing
    parent_folder_path = './solutions/'
    folder_path = os.path.join(parent_folder_path, model_name)
    if not model_name in os.listdir(parent_folder_path):
        os.mkdir(folder_path)
    pickle_name = f'sol_{model_name}_{spec_string}.pkl'
    pickle_path = os.path.join(folder_path, pickle_name)

    # Solve or load solution
    sol = None
    if pickle_path not in os.listdir(folder_path) or force_recalculation:
        sol = solve_bvp(
            ode,
            boundary_condition,
            x_axis,
            y,
            max_nodes=1000000,
            verbose=verbose)
        with open(pickle_path, 'wb') as file:
            pickle.dump(sol, file)
        if verbose:
            print(f'ODE problem solved and saved under {pickle_path}.')
    else:
        if verbose:
            print(f'File {pickle_name} found.')
        with open(pickle_path, 'rb') as file:
            sol = pickle.load(file)

    return sol

def ode_guy_chapman(x, y):
    """
    System of dimensionless 1st order ODE's that we solve
    x: dimensionless x-axis of length n, i.e. kappa (1/m) times x-position (m).
    y: dimensionless potential phi and dphi/dx, size 2 x n.
    """
    dy1 = y[1, :]
    dy2 = np.sinh(y[0, :])

    return np.vstack([dy1, dy2])
