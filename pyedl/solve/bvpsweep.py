"""
Implementation of parameter sweep for boundary value problems
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d


def interpolate_sol_list(
    x0: np.ndarray,
    sol_list: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Interpolate a list of solutions to correspond to the initial mesh x0
    Returns: array of shape (n, l, m)
    """
    n = sol_list[0][1].shape[0]  # dimension of y
    y_array = np.zeros((n, len(sol_list), len(x0)))

    for i, (x, y) in enumerate(sol_list):
        for j in range(n):
            y_array[j, i, :] = interp1d(x, y[j, :])(x0)

    return y_array


def sweep_from_initial_guess(
    fun: callable,
    bc: callable,
    x0: np.ndarray,
    y0: np.ndarray,
    sweep_par: np.ndarray,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Parameter sweep of solve_bvp, starting at an initial guess and using
    the solution as the initial guess for the next parameter value.

    Returns: maximum relative residual, array of interpolated solutions
    of shape (n, l, m)
    """

    def bc_lambda(p):
        return lambda ya, yb: bc(ya, yb, p)

    x = x0
    y = y0
    sol_list = []
    max_residuals = []
    mesh_changed = False

    for p in sweep_par:
        sol = solve_bvp(
            fun,
            bc_lambda(p),
            x,
            y,
            tol=tol,
            max_nodes=int(1e8),
            verbose=0,
        )
        sol_list.append((sol.x, sol.y))
        max_residuals.append(np.max(sol.rms_residuals))

        if len(sol.x) > len(x):
            mesh_changed = True

        x = sol.x
        y = sol.y

    if mesh_changed:
        print(
            "Warning: solve_bvp has increased the mesh density. "
            + "Details may be lost when interpolating back to x0. "
            + "To avoid this, increase the initial mesh density. "
        )
    return np.max(max_residuals), interpolate_sol_list(x0, sol_list)


def sweep_solve_bvp(
    fun: callable,
    bc: callable,
    x0: np.ndarray,
    y0: np.ndarray,
    sweep_par: np.ndarray,
    sweep_par_start: float,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    This function performs a parameter sweep for a boundary value problem,
    where the boundary condition depends on a parameter p:

        dy / dx = f(x, y), a <= x <= b
        bc(y(a), y(b), p) = 0

    The initial guess y0 is specified at the value sweep_par_start in the
    parameter sweep. By using the solution at one parameter value as initial
    condition for the next, difficult boundary condition problems can be solved.
    The more finely spaced sweep_par is, the better the initial guesses are.

    See also the documentation for Scipy's ``solve_bvp`` function.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(x, y)``.
        All arguments are ndarray: ``x`` with shape (m,), ``y`` with shape (n, m),
        meaning that ``y[:, i]`` corresponds to ``x[i]``. The
        return value must be an array with shape (n, m) and with the same
        layout as ``y``.
    bc : callable
        Function evaluating residuals of the boundary conditions. The calling
        signature is ``bc(ya, yb, p)`` where p is the parameter that is
        swept over. All arguments are ndarray: ``ya`` and ``yb`` with shape (n,),
        and ``p`` a scalar. The return value must be an array with
        shape (n,).
    x0 : array_like, shape (m,)
        Initial mesh. Must be a strictly increasing sequence of real numbers
        with ``x[0]=a`` and ``x[-1]=b``.
    y0 : array_like, shape (n, m)
        Initial guess for the function values at the mesh nodes.
    sweep_par : array_like, shape (l,)
        Values of p to sweep over. Must be strictly monotonous.
    sweep_par_start: float
        Value of p where the initial guess y0 is specified.
    tol : float, optional
        Desired tolerance of the solution. See documentation for scipy solve_bvp.
        Default is 1e-3.


    Returns
    -------
    y: ndarray, shape (n, l, m)
        Solution values at the initial mesh nodes, for all parameter values.
        If ``solve_bvp`` has expanded the mesh, the final solution is interpolated
        back to the initial mesh, and a warning message is printed.
    """
    assert sweep_par.ndim == 1, "sweep_par should be one-dimensional"
    assert np.all(np.diff(sweep_par) > 0) or np.all(
        np.diff(sweep_par) < 0
    ), "sweep_par should be strictly monotonous"
    assert sweep_par_start >= np.min(sweep_par) and sweep_par_start <= np.max(
        sweep_par
    ), "sweep_par_start must be in the range specified by sweep_par"

    # Find starting index of sweep
    i_start = np.argmin(np.abs(sweep_par)).squeeze()

    neg_res, neg_sweep = sweep_from_initial_guess(
        fun,
        bc,
        x0,
        y0,
        sweep_par[i_start::-1],
        tol,
    )
    pos_res, pos_sweep = sweep_from_initial_guess(
        fun,
        bc,
        x0,
        y0,
        sweep_par[i_start::1],
        tol,
    )

    print(
        f"Sweep from {sweep_par[0]:.2e} to {sweep_par[-1]:.2e}, "
        + f"starting at {sweep_par_start:.2e}. "
        + f"Maximum relative residual: {max(neg_res, pos_res):.3e}."
    )

    return np.concatenate(
        (neg_sweep[:, ::-1, :], pos_sweep[:, 1:, :]),
        axis=1,
    )
