# Example implementation of correlation function for isolated
# spheres with log-normal size distribution. Including comparison
# with the original lmfit package.

import sys
import time
import numpy as np
import cpu_lmfit
import lmfit as lm_pkg


def G_term(z, d):
    if z == 0.0:
        return 1.0  # Edge case: G(0) = 1

    z_over_d = z / d

    # Term1 calculation
    term1 = 0.0
    if z < d:
        sqrt_term = np.sqrt(1.0 - z_over_d * z_over_d)
        term1 = sqrt_term * (1.0 + 0.5 * z_over_d * z_over_d)

    # Term2 calculation
    term2 = 0.0
    if z < d:
        prefactor = 2.0 * z_over_d * z_over_d * (1.0 - z_over_d * z_over_d)
        denominator = 1.0 + np.sqrt(1.0 - z_over_d * z_over_d)
        # Guard against invalid log argument
        # If z_over_d / denominator <= 0, math.log will fail; in C++ code logf would produce -inf; keep behavior safe
        term2 = prefactor * np.log(z_over_d / denominator) if z_over_d / denominator > 0.0 else 0.0

    result = term1 + term2
    return result if result > 0.0 else 0.0


def lognormal_distribution(x, s):
    # x is the normalized variable (y in C++ code), s is shape parameter
    # if x <= 0.0:
    #     return 0.0
    exponent = -0.5 * (np.log(x) / s) * (np.log(x) / s)
    denominator = s * x * np.sqrt(2.0 * np.pi)
    return np.exp(exponent) / denominator if denominator > 1e-300 else 0.0


def get_lognormal_distribution(s, scale, loc, N_LOGNORMAL_POINTS = 1000):
    # Generates log-normal distribution points and values using numpy
    mean = np.log(scale)
    stddev = s

    # Quantiles for the underlying normal distribution
    normal_lo = mean - 3.09023 * stddev  # 0.1% quantile
    normal_hi = mean + 3.09023 * stddev  # 99.9% quantile

    lo_lim = loc + np.exp(normal_lo)
    hi_lim = loc + np.exp(normal_hi)

    distr_range = np.linspace(lo_lim, hi_lim, N_LOGNORMAL_POINTS)
    y = (distr_range - loc) / scale
    # Vectorized PDF computation
    distr_values = np.where(y > 0.0, np.exp(-0.5 * (np.log(y) / s) ** 2) / (s * y * np.sqrt(2.0 * np.pi)), 0.0) / scale

    return distr_range, distr_values


def corr_isol_spheres_polydisp_lognorm(xi, p):
    s = p[0]
    scale = p[1]
    loc = p[2]

    distr_range, distr_values = get_lognormal_distribution(s, scale, loc)
    sum_probs = np.sum(distr_values)
    
    result = 0.0
    for idx, d_val in enumerate(distr_range):
        if d_val > 0.0:
            result += distr_values[idx] * G_term(xi, d_val)

    return result / float(sum_probs)


def create_correlation_data(true_params, n_points, noise_level):
    lower_bound = 0.0
    upper_bound = 100.0
    x_data = np.zeros(n_points, dtype=float)
    y_data = np.zeros(n_points, dtype=float)
    for i in range(n_points):
        x_data[i] = lower_bound + i * (upper_bound - lower_bound) / (n_points - 1)
        y_data[i] = corr_isol_spheres_polydisp_lognorm([x_data[i]], true_params) + noise_level * (np.random.random() - 0.5)

    return x_data, y_data


def fit():
    n_points = 100
    n_params = 3

    true_params = [0.2, 10.0, 2.0]  # s, scale, loc
    x_data, y_data = create_correlation_data(true_params, n_points, 0.01)

    initial_params = [1, 1, 1]

    # --- Run the original cpu_lmfit solver and time it ---
    cpu_lm = cpu_lmfit.LMFit()
    cpu_lm.ensure_capacity(n_points, n_params)

    t0 = time.perf_counter()
    success_cpu = cpu_lm.levenberg_marquardt_fit(x_data, y_data, initial_params, corr_isol_spheres_polydisp_lognorm, None, 1e-6, 100, 0.01)
    t1 = time.perf_counter()
    cpu_time = t1 - t0

    if success_cpu:
        print("cpu_lmfit: fit succeeded (analytical-derivative). Time: {:.6f} s".format(cpu_time))
        try:
            # Print any available metrics (original behaviour)
            cpu_lm.print_fit_metrics()
        except Exception:
            pass
    else:
        print("cpu_lmfit: fitting failed. Time: {:.6f} s".format(cpu_time))

    # --- Now run the lmfit package (Levenberg-Marquardt via 'leastsq') ---
    # Build a residual function that computes model values for all x points
    def residual(params, x, data):
        pvals = [params['s'].value, params['scale'].value, params['loc'].value]
        model = np.array([corr_isol_spheres_polydisp_lognorm(xi, pvals) for xi in x])
        # residual = data - model
        return data - model

    params = lm_pkg.Parameters()
    params.add('s', value=float(initial_params[0]))
    params.add('scale', value=float(initial_params[1]))
    params.add('loc', value=float(initial_params[2]))

    minimizer = lm_pkg.Minimizer(residual, params, fcn_args=(x_data, y_data))

    t2 = time.perf_counter()
    result = minimizer.minimize(method='leastsq', max_nfev=1000)
    t3 = time.perf_counter()
    lmfit_time = t3 - t2

    # Print summary for lmfit
    print('\nlmfit (third-party) results:')
    print('  Time: {:.6f} s'.format(lmfit_time))
    try:
        s_val = result.params['s'].value
        scale_val = result.params['scale'].value
        loc_val = result.params['loc'].value
        print('s     =', s_val)
        print('scale =', scale_val)
        print('loc   =', loc_val)
    except Exception as e:
        print('Could not extract lmfit parameters:', e)

    # Print fit report
    try:
        print('\nFull lmfit report:')
        print(lm_pkg.fit_report(result))
    except Exception:
        pass

    # Return both success flags and timings for possible programmatic use
    return {'cpu_success': success_cpu, 'cpu_time': cpu_time, 'lmfit_result': result, 'lmfit_time': lmfit_time}


if __name__ == '__main__':
    try:
        fit()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
