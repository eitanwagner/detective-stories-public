"""Standalone utility functions for detective story analysis."""

import numpy as np
from scipy.stats import beta


def clopper_pearson_interval(successes, trials, alpha=0.32):
    """
    Calculate Clopper-Pearson confidence interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of successes
    trials : int
        Total number of trials
    alpha : float
        Significance level (default 0.32 for ~1 std equivalent, 68% CI for visualization)

    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if trials == 0:
        return (0.0, 0.0)

    if successes == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, successes, trials - successes + 1)

    if successes == trials:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)

    return (lower, upper)


def argmax_distribution(dist):
    """Convert each row of a distribution matrix to a hard argmax.

    For each row the maximum value is found.  If that maximum is unique the
    full probability mass (1.0) is placed on that index.  When multiple indices
    tie for the maximum the mass is split equally among them.

    Parameters
    ----------
    dist : np.ndarray, shape (T, K) or (K,)
        A 2-D array of per-timestep distributions (rows must sum to ~1).
        A 1-D array (single timestep) is also accepted and will be returned
        as 2-D with shape (1, K).

    Returns
    -------
    np.ndarray, shape (T, K)
        Hard (or tied) argmax distribution, always 2-D.
    """
    dist = np.atleast_2d(dist)  # (K,) -> (1, K); (T, K) unchanged
    result = np.zeros_like(dist, dtype=float)
    for i, row in enumerate(dist):
        max_val = np.max(row)
        max_mask = (row == max_val)
        result[i] = max_mask / max_mask.sum()
    return result
