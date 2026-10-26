"""Monte Carlo estimation of detection and false-alarm probabilities."""

import numpy as np

from .detector import energy_statistic
from .signals import received_signal


def monte_carlo_probability(threshold, snr_linear, n_samples, n_trials, primary_present, rng):
    """Estimate P(T > threshold) under H0 or H1 by Monte Carlo.

    Vectorized: one (n_trials, n_samples) draw per call.
    """
    noise_power = 1.0 / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=(n_trials, n_samples))
    if primary_present:
        signal = rng.choice([-1.0, 1.0], size=(n_trials, n_samples))
        y = signal + noise
    else:
        y = noise
    stats = np.mean(y**2, axis=1)
    return float(np.mean(stats > threshold))


def monte_carlo_cooperative(threshold, snr_linear, n_samples, n_trials, k, primary_present, rng):
    """Estimate the OR-fusion probability with K independent nodes."""
    decisions = np.zeros(n_trials, dtype=bool)
    for _ in range(k):
        noise_power = 1.0 / snr_linear
        noise = rng.normal(0.0, np.sqrt(noise_power), size=(n_trials, n_samples))
        if primary_present:
            signal = rng.choice([-1.0, 1.0], size=(n_trials, n_samples))
            y = signal + noise
        else:
            y = noise
        stats = np.mean(y**2, axis=1)
        decisions |= stats > threshold
    return float(np.mean(decisions))
