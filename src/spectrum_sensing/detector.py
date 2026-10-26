"""Energy detector: non-coherent, needs no knowledge of the primary signal."""

import numpy as np
from scipy.stats import norm


def energy_statistic(y):
    """Test statistic T = (1/N) * sum |y[n]|^2."""
    y = np.asarray(y)
    return float(np.mean(np.abs(y) ** 2))


def threshold_for_pfa(pfa, noise_power, n_samples):
    """Decision threshold giving false-alarm probability `pfa` under H0.

    For large N the statistic is approximately Gaussian by the CLT:
    T | H0 ~ Normal(noise_power, 2 * noise_power^2 / N).
    """
    std_h0 = noise_power * np.sqrt(2.0 / n_samples)
    return noise_power + std_h0 * norm.ppf(1.0 - pfa)


def detect(y, threshold):
    """Return True if the band is declared busy (H1), else False."""
    return energy_statistic(y) > threshold
