"""Closed-form Gaussian-approximation performance of the energy detector."""

import numpy as np
from scipy.stats import norm


def snr_db_to_linear(snr_db):
    return 10.0 ** (np.asarray(snr_db, dtype=float) / 10.0)


def theoretical_pfa(threshold, noise_power, n_samples):
    """Pfa = Q((gamma - sigma^2) / (sigma^2 * sqrt(2/N)))."""
    std_h0 = noise_power * np.sqrt(2.0 / n_samples)
    return 1.0 - norm.cdf((threshold - noise_power) / std_h0)


def theoretical_pd(threshold, snr_linear, n_samples):
    """Pd under H1 for BPSK (unit power) in AWGN.

    Under H1, y ~ N(s, sigma^2) with E[y^2] = 1 + sigma^2 and
    Var(y^2) = 2*sigma^4 + 4*sigma^2 (since s^2 = 1), so
    T | H1 ~ Normal(1 + sigma^2, (2*sigma^4 + 4*sigma^2) / N).
    """
    noise_power = 1.0 / snr_linear
    mean_h1 = 1.0 + noise_power
    std_h1 = np.sqrt((2.0 * noise_power**2 + 4.0 * noise_power) / n_samples)
    return 1.0 - norm.cdf((threshold - mean_h1) / std_h1)
