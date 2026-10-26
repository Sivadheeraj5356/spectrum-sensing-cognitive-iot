import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectrum_sensing.signals import received_signal
from spectrum_sensing.detector import energy_statistic, threshold_for_pfa, detect

N = 1000
PFA = 0.1


def test_energy_statistic_white_noise_mean():
    rng = np.random.default_rng(0)
    y = received_signal(200_000, snr_linear=1.0, primary_present=False, rng=rng)
    assert energy_statistic(y) == pytest.approx(1.0, abs=0.02)


def test_threshold_achieves_target_pfa():
    rng = np.random.default_rng(1)
    lam = threshold_for_pfa(PFA, noise_power=1.0, n_samples=N)
    trials = 20_000
    alarms = sum(detect(received_signal(N, 1.0, False, rng), lam) for _ in range(trials))
    assert alarms / trials == pytest.approx(PFA, abs=0.015)


def test_detect_positive_snr_high_pd():
    rng = np.random.default_rng(2)
    snr = 10.0  # linear; noise power = 0.1
    lam = threshold_for_pfa(PFA, noise_power=1.0 / snr, n_samples=N)
    hits = sum(detect(received_signal(N, snr, True, rng), lam) for _ in range(500))
    assert hits / 500 > 0.99
