"""
Implementation of the Langevin function and related functions
that appear in the statistical mechanics of dipolar media
"""

import numpy as np


def langevin_x_over_x(x):  # pylint: disable=invalid-name
    """
    Returns L(x)/x, where L(x) is the Langevin function.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ndarray
        L(x)/x, where L(x) = 1/tanh(x) - 1/x. For small x, the function value is 1/3.
    """
    x = np.atleast_1d(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (1 / np.tanh(x) - 1 / x) / x

    small_x_mask = np.abs(x) <= 1e-4
    result[small_x_mask] = 1 / 3 - x[small_x_mask] ** 2 / 45

    return result


def langevin_x(x):  # pylint: disable=invalid-name
    """
    Returns L(x), where L(x) is the Langevin function.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ndarray
        L(x), where L(x) = 1/tanh(x) - 1/x. For small x, the function value is zero.
    """
    x = np.atleast_1d(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 1 / np.tanh(x) - 1 / x

    small_x_mask = np.abs(x) <= 1e-4
    result[small_x_mask] = x[small_x_mask] / 3 - x[small_x_mask] ** 3 / 45

    return result


def d_langevin_x(x):  # pylint: disable=invalid-name
    """
    Returns the derivative of the Langevin function with respect to x.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ndarray
        dL/dx = 1/x^2 - 1/sinh^2(x). For small x, the function value is 1/3.
    """
    x = np.atleast_1d(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = 1 / x**2 - 1 / np.sinh(x) ** 2

    small_x_mask = np.abs(x) < 1e-4
    result[small_x_mask] = 1 / 3 - x[small_x_mask] ** 2 / 15
    large_x_mask = np.isinf(result)
    result[large_x_mask] = 0

    return result


def sinh_x_over_x(x):
    """
    Returns sinh(x)/x.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ndarray
        sinh(x)/x. For small x, the function value is one. For very large x,
        returns the maximum float representation.
    """
    x = np.atleast_1d(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.sinh(x) / x

    small_x_mask = np.abs(x) < 1e-4
    large_x_mask = np.isinf(result)

    max_float = np.finfo(np.float64).max
    result[small_x_mask] = 1 + x[small_x_mask] ** 2 / 6
    result[large_x_mask] = max_float

    return result
