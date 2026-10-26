# Spectrum Sensing and Cognitive IoT Network

A Python simulation study of **cooperative energy-detection spectrum sensing** for
cognitive radio IoT (CR-IoT) networks. B.Tech ECE project, IIIT Surat.

IoT devices (secondary users) check whether a licensed band is idle before using it.
Each node measures the received signal energy over N = 1000 samples and compares it
with a threshold set for a false-alarm probability of 0.1. Several nodes (1, 3 and 5)
send 1-bit decisions to a fusion center that applies the **OR rule**: the band is
declared busy if any node says busy.

## System model

Binary hypothesis test on N received samples y[n]:

- **H0** (primary absent): `y[n] = w[n]`
- **H1** (primary present): `y[n] = s[n] + w[n]`

`s[n]` is BPSK (+1/-1 random symbols), `w[n]` is AWGN. SNR is swept from -20 to +5 dB,
with 2000 Monte Carlo trials per point.

## Algorithm

Energy detection (non-coherent, no knowledge of the primary signal needed):

- Test statistic: `T = (1/N) * sum |y[n]|^2`
- Decide H1 if `T > threshold`
- Threshold is set from the noise-only distribution so that `Pfa = 0.1`
- For large N, T is approximately Gaussian (central limit theorem), giving the
  theoretical Pd curve the simulation is validated against

## Repository layout

```
src/spectrum_sensing/   the detector as an importable package
  signals.py            BPSK + AWGN signal model
  detector.py           energy statistic, threshold, decision
  theory.py             closed-form Gaussian-approximation Pd / Pfa
  cooperative.py        OR fusion and its closed-form Pd / Pfa
  simulation.py         vectorized Monte Carlo estimators
scripts/run_simulation.py   runs everything, writes results/ (3 plots + results.json)
tests/                  pytest suite (10 tests): detector, theory-vs-simulation, OR rule
results/                generated plots and numbers
docs/REPORT.md          full project report
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/run_simulation.py          # seed 42, 2000 trials per point (default)
python -m pytest tests/ -q                # 10 tests
```

## Headline results (seed 42, 2000 trials/point, Pfa = 0.1)

Detection probability at **-10 dB**:

| Nodes (OR rule) | Simulation | Theory |
|---|---|---|
| 1 | 0.795 | 0.808 |
| 3 | 0.995 | 0.993 |
| 5 | 1.000 | 1.000 |

The OR rule raises detection at low SNR at the cost of a higher overall false-alarm
rate (simulated Pfa: 0.098 / 0.271 / 0.404 for 1/3/5 nodes; theory 0.1 / 0.271 / 0.410).

The three plots (Pd vs SNR, ROC at -10 dB, cooperative OR comparison) are not
checked in as images; run `python scripts/run_simulation.py` to regenerate them
into `results/` along with `results.json`, which holds every number behind them.

## Limitations and future work

Energy detection degrades at very low SNR and hits the "SNR wall" under noise-power
uncertainty; it cannot distinguish the primary signal from other interference; the
baseline is AWGN only (no fading or shadowing); the OR rule raises false alarms as
nodes are added. Next steps: Rayleigh fading, AND/majority fusion, noise uncertainty,
an ML-based detector, SNR-weighted fusion.

## References

1. Miah, Schukat, Barrett, Cespedes, Armada, "Enhanced spectrum sensing for AI-enabled
   cognitive radio IoT with noise uncertainty," *ITU Journal on Future and Evolving
   Technologies*, vol. 6, no. 1, March 2025. Main reference; our project is the
   simpler energy-detection baseline of that setting and does not reproduce their
   enhanced-KLD method.
2. Wu, Ng, Lam, "Energy-Efficient Cooperative Spectrum Sensing Using Machine Learning
   Algorithm," *Sensors*, 2022.
3. Wu et al., "Distributed Sequential Detection for Cooperative Spectrum Sensing in
   Cognitive Internet of Things," *Sensors*, 2024.
4. Dissanayake et al., "Uniform Planar Array Based Weighted Cooperative Spectrum
   Sensing for Cognitive Radio Networks," arXiv:2504.10034, 2025.
