import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d


def weighted_distribution_matching(A, A_prime, bandwidth=1.0, num_points=1000):
    """
    Soft Scaling using KDE and weighted matching based on the density of A.
    Maps values in A_prime to the distribution of A by considering the density of A.
    """
    # Estimate the kernel density of A
    kde_A = gaussian_kde(A, bw_method=bandwidth)
    x_vals_A = np.linspace(np.min(A), np.max(A), num_points)
    density_A = kde_A.evaluate(x_vals_A)

    # Normalize density to [0, 1]
    density_A /= np.max(density_A)

    # Compute cumulative distribution function (CDF) for A
    cdf_A = np.cumsum(density_A) / np.sum(density_A)

    # Interpolation function for matching A_prime's values based on A's CDF
    f_map_A = interp1d(cdf_A, x_vals_A, bounds_error=False, fill_value=(np.min(A), np.max(A)))

    # Map the points in A_prime based on A's CDF
    cdf_A_prime = np.linspace(0, 1, len(A_prime))
    A_prime_mapped = f_map_A(cdf_A_prime)

    return A_prime_mapped


def truncated_quantile_mapping(A, A_prime, lower_pct=5, upper_pct=95):
    # Compute target quantiles in A
    lower_bound = np.percentile(A, lower_pct)
    upper_bound = np.percentile(A, upper_pct)

    # Restrict A within bounds
    A_truncated = A[(A >= lower_bound) & (A <= upper_bound)]

    # Compute CDFs
    A_cdf = np.linspace(0, 1, len(A_truncated))
    A_prime_cdf = np.linspace(0, 1, len(A_prime))

    # Sort for interpolation
    A_sorted = np.sort(A_truncated)
    A_prime_sorted = np.sort(A_prime)

    # Map A_prime to A using quantile mapping
    f_map = interp1d(A_prime_cdf, A_prime_sorted, bounds_error=False, fill_value=(A_prime.min(), A_prime.max()))
    A_prime_mapped = np.interp(A_prime_cdf, A_cdf, A_sorted)

    return A_prime_mapped


def density_based_mapping(A, A_prime, bandwidth=1.0):
    # Estimate KDE for A
    kde = gaussian_kde(A, bw_method=bandwidth)
    kde_cdf = np.cumsum(kde.evaluate(np.sort(A)))
    kde_cdf /= kde_cdf[-1]  # Normalize to [0, 1]

    # Interpolation to map A_prime
    A_sorted = np.sort(A)
    f_map = interp1d(kde_cdf, A_sorted, bounds_error=False, fill_value=(A.min(), A.max()))

    # Evaluate mapping for A_prime
    A_prime_cdf = np.linspace(0, 1, len(A_prime))
    A_prime_mapped = f_map(A_prime_cdf)

    return A_prime_mapped

