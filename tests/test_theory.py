import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectrum_sensing.detector import threshold_for_pfa
from spectrum_sensing.theory import theoretical_pd, theoretical_pfa, snr_db_to_linear
from spectrum_sensing.simulation import monte_carlo_probability

N = 1000
PFA = 0.1


def test_theory_inverts_threshold():
    lam = threshold_for_pfa(PFA, noise_power=1.0, n_samples=N)
    assert theoretical_pfa(lam, 1.0, N) == pytest.approx(PFA, abs=1e-10)


def test_snr_conversion():
    assert snr_db_to_linear(0.0) == pytest.approx(1.0)
    assert snr_db_to_linear(-10.0) == pytest.approx(0.1)
    assert snr_db_to_linear(10.0) == pytest.approx(10.0)


def test_theory_matches_simulation_within_monte_carlo_error():
    rng = np.random.default_rng(3)
    for snr_db in (-15.0, -10.0, -5.0):
        g = snr_db_to_linear(snr_db)
        lam = threshold_for_pfa(PFA, 1.0 / g, N)
        sim = monte_carlo_probability(lam, g, N, 20_000, True, rng)
        th = theoretical_pd(lam, g, N)
        assert sim == pytest.approx(th, abs=0.02)
