"""Signal model: BPSK primary user signal in additive white Gaussian noise."""

import numpy as np


def bpsk_signal(n_samples, rng):
    """Random BPSK symbols (+1/-1), shape (n_samples,)."""
    return rng.choice([-1.0, 1.0], size=n_samples)


def awgn(n_samples, noise_power, rng):
    """Real additive white Gaussian noise with the given power (variance)."""
    return rng.normal(0.0, np.sqrt(noise_power), size=n_samples)


def received_signal(n_samples, snr_linear, primary_present, rng):
    """Received samples under H0 (noise only) or H1 (BPSK + noise).

    Signal power is normalized to 1, so noise power = 1 / snr_linear.
    """
    noise_power = 1.0 / snr_linear
    noise = awgn(n_samples, noise_power, rng)
    if primary_present:
        return bpsk_signal(n_samples, rng) + noise
    return noise
