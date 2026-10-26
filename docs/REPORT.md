# Cooperative Energy-Detection-Based Spectrum Sensing for Cognitive IoT Networks: A Python Simulation Study

B.Tech ECE project, IIIT Surat. Project title: **Spectrum Sensing and Cognitive IoT Network**.

## 1. Problem

IoT deployments are growing fast and the unlicensed bands they use are congested.
Licensed (primary) users leave their bands idle much of the time. Cognitive radio
lets IoT devices (secondary users) use a licensed band opportunistically, but only
if they can reliably detect whether the primary user is active. Wrong decisions
either interfere with the licensed user (missed detection) or waste free spectrum
(false alarm).

## 2. Objective

Design and evaluate an energy detector for secondary IoT nodes, and show how
cooperative sensing among several nodes improves detection reliability.

## 3. System model

Binary hypothesis test on N received samples y[n]:

- H0 (primary absent): `y[n] = w[n]`
- H1 (primary present): `y[n] = s[n] + w[n]`

`s[n]` = BPSK (+1/-1 random symbols), `w[n]` = AWGN. N = 1000 samples per sensing
interval, SNR swept from -20 to +5 dB in 1 dB steps.

## 4. Algorithm

Energy detection (non-coherent, needs no knowledge of the primary signal):

- Test statistic `T = (1/N) * sum |y[n]|^2`
- Decide H1 if `T > threshold`, else H0
- The threshold is set from the noise-only distribution so that the false-alarm
  probability is fixed at Pfa = 0.1. For large N, T is approximately Gaussian
  (central limit theorem), which gives the theoretical Pd curve:
  `Pd = Q((threshold - mean_H1) / std_H1)`
- Cooperative sensing: K nodes (3 and 5) sense independently, each sends a 1-bit
  decision to a fusion center. OR rule: declare busy if any node says busy.

## 5. Method / framework

Python, numpy for signals and Monte Carlo, matplotlib for plots, scipy for the
Gaussian tail functions, pytest for the test suite. 2000 Monte Carlo trials per
point, seed 42 for reproducibility. No external toolboxes.

Package layout: `src/spectrum_sensing/` (signals, detector, theory, cooperative,
simulation), runner in `scripts/run_simulation.py`, tests in `tests/`.

## 6. Results

All numbers below are produced by `scripts/run_simulation.py` (seed 42, 2000 trials
per point) and stored in `results/results.json`.

**Plot 1 - Pd vs SNR (`results/pd_vs_snr.png`).** Single-node simulation overlaid
with the Gaussian-approximation theory. The simulation tracks the theory across the
whole -20 to +5 dB sweep, which validates the model.

**Plot 2 - ROC at SNR = -10 dB (`results/roc_minus10db.png`).** Pd vs Pfa for 1, 3
and 5 nodes. The cooperative curves sit strictly above the single-node curve.

**Plot 3 - Single node vs cooperative OR (`results/cooperative_or.png`).**
Detection probability vs SNR for 1, 3 and 5 nodes.

Detection probability at selected SNR values (simulation):

| SNR (dB) | 1 node | 3 nodes (OR) | 5 nodes (OR) |
|---|---|---|---|
| -20 | 0.143 | 0.388 | 0.548 |
| -15 | 0.293 | 0.629 | 0.801 |
| -12 | 0.557 | 0.905 | 0.983 |
| -10 | 0.795 | 0.995 | 1.000 |
| -8 | 0.978 | 1.000 | 1.000 |
| -5 | 1.000 | 1.000 | 1.000 |

Findings:

- Pd rises with SNR, as expected, and matches the Gaussian-approximation theory.
- Cooperation raises detection noticeably at low SNR: at -10 dB a single node
  detects with probability about 0.80, three nodes reach about 0.99 and five nodes
  reach 1.0.
- The cost of the OR rule is a higher overall false-alarm rate: simulated Pfa is
  0.098, 0.271 and 0.404 for 1, 3 and 5 nodes (closed form 0.1, 0.271, 0.410).

## 7. Limitations

- Energy detection degrades at very low SNR and suffers the "SNR wall" under
  noise-power uncertainty.
- It cannot distinguish the primary signal from other interference.
- AWGN only; no fading or shadowing in the baseline.
- The OR rule increases false alarms as K grows.

## 8. Related work / reference

Miah, Schukat, Barrett, Cespedes, Armada, "Enhanced spectrum sensing for AI-enabled
cognitive radio IoT with noise uncertainty," *ITU Journal on Future and Evolving
Technologies*, vol. 6, issue 1, March 2025. It uses BPSK and Gaussian noise,
compares against energy detection, and uses cooperative IoT nodes. Our work is the
simpler baseline of that setting; we do not claim to reproduce their enhanced-KLD
method.

## 9. Future work

Add Rayleigh fading, AND/majority fusion rules, noise uncertainty, and an ML-based
detector; weight nodes by SNR instead of plain OR.

## 10. 30-second pitch

"IoT devices share a crowded radio band, while licensed bands often sit idle. We
simulate an energy detector that lets an IoT node decide if a licensed band is free,
evaluate it with detection probability, false alarm and ROC curves, and show that
cooperation among several nodes improves reliability."
