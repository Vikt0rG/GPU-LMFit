# Python port of examples/CPU/Gauss_1D.cpp using the package in package/

import sys
import os
import math
import random
import numpy as np

# Import the installed compiled extension module directly (cpu_lmfit)
import cpu_lmfit


def gauss_func(x, p):
    A = p[0]
    mu = p[1]
    sigma = p[2]
    z = (x - mu) / sigma
    return A * math.exp(-0.5 * z * z)


def gauss_deriv(x, p):
    A = p[0]
    mu = p[1]
    sigma = p[2]
    z = (x - mu) / sigma
    e = math.exp(-0.5 * z * z)
    return np.array([e, A * e * (z / sigma), A * e * (z * z / sigma)], dtype=float)


def create_gaussian_data(true_params, n_points, noise_level):
    lower_bound = -5.0
    upper_bound = 5.0
    x_data = np.linspace(lower_bound, upper_bound, n_points)
    y = np.empty(n_points, dtype=float)
    for i in range(n_points):
        y[i] = gauss_func(x_data[i], true_params) + noise_level * (random.random() - 0.5)
    return x_data, y


def fit():
    n_points = 100
    n_params = 3

    true_params = np.array([6.0, 0.0, 1.0], dtype=float)
    x_data, y_data = create_gaussian_data(true_params, n_points, 0.1)

    initial_params = np.array([3.0, 1.0, 2.0], dtype=float)

    lm = cpu_lmfit.LMFit()
    # Wrapper expects numpy arrays and Python callables. We pass a model function that
    # accepts (x_scalar, params_array) and returns a scalar. For derivative, pass function
    # that returns array-like of length n_params.

    success = lm.levenberg_marquardt_fit(x_data, y_data, initial_params, gauss_func, gauss_deriv, 1e-6, 100, 0.01)
    if success:
        print("Analytic-derivative fit metrics:")
        lm.print_fit_metrics()
    else:
        print("Fitting failed.")

    # Numeric derivative fit
    initial_params2 = np.array([3.0, 1.0, 2.0], dtype=float)
    success2 = lm.levenberg_marquardt_fit(x_data, y_data, initial_params2, gauss_func, None, 1e-6, 100, 0.01)
    print("\nNumeric-derivative fit results:")
    if success2:
        lm.print_fit_metrics()
    else:
        print("Numeric fit failed.")

    return success


if __name__ == '__main__':
    try:
        fit()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
