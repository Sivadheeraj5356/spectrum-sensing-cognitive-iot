import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectrum_sensing.cooperative import or_fusion, cooperative_pd, cooperative_pfa
from spectrum_sensing.simulation import monte_carlo_cooperative
from spectrum_sensing.detector import threshold_for_pfa


def test_or_fusion_logic():
    assert or_fusion([False, False, True])
    assert not or_fusion([False, False, False])


def test_cooperative_closed_form():
    assert cooperative_pd(0.5, 2) == pytest.approx(0.75)
    assert cooperative_pfa(0.1, 3) == pytest.approx(1 - 0.9**3)


def test_or_rule_never_worse_than_single_node():
    rng = np.random.default_rng(4)
    lam = threshold_for_pfa(0.1, noise_power=10.0, n_samples=1000)
    single = monte_carlo_cooperative(lam, 0.1, 1000, 10_000, 1, True, rng)
    coop = monte_carlo_cooperative(lam, 0.1, 1000, 10_000, 5, True, rng)
    assert coop >= single


def test_or_rule_matches_closed_form():
    rng = np.random.default_rng(5)
    lam = threshold_for_pfa(0.1, noise_power=1.0, n_samples=1000)
    pfa_sim = monte_carlo_cooperative(lam, 1.0, 1000, 10_000, 3, False, rng)
    assert pfa_sim == pytest.approx(cooperative_pfa(0.1, 3), abs=0.02)
