"""Run the full spectrum-sensing simulation and produce the three project plots.

Usage:
    python scripts/run_simulation.py [--trials 2000] [--seed 42] [--outdir results]

Produces:
    pd_vs_snr.png        Pd vs SNR, single node, simulation overlaid with theory
    roc_minus10db.png    ROC curve (Pd vs Pfa) at SNR = -10 dB, 1/3/5 nodes
    cooperative_or.png   Pd vs SNR for 1, 3 and 5 nodes under the OR rule
    results.json         every simulated and theoretical number behind the plots
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from spectrum_sensing import (  # noqa: E402
    threshold_for_pfa, theoretical_pd, snr_db_to_linear,
    monte_carlo_probability, monte_carlo_cooperative,
)
from spectrum_sensing.simulation import cooperative_statistics  # noqa: E402
from spectrum_sensing.cooperative import cooperative_pd  # noqa: E402

N_SAMPLES = 1000
PFA_TARGET = 0.1
SNR_DB = np.arange(-20.0, 6.0, 1.0)
NODES = (1, 3, 5)
ROC_SNR_DB = -10.0
ROC_POINTS = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000, help="Monte Carlo trials per point")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducibility)")
    ap.add_argument("--outdir", default="results", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    out = {"config": {"n_samples": N_SAMPLES, "n_trials": args.trials,
                      "pfa_target": PFA_TARGET, "seed": args.seed,
                      "snr_db": SNR_DB.tolist(), "nodes": list(NODES)}}

    # Per-node threshold from the target false-alarm probability (noise power 1).
    # In this normalization the signal power equals the linear SNR and sigma^2 = 1,
    # matching the recovered Oct 5 script. Equivalent to unit-signal normalization.
    snr_lin = snr_db_to_linear(SNR_DB)

    def threshold(snr_linear):
        noise_power = 1.0 / snr_linear
        return threshold_for_pfa(PFA_TARGET, noise_power, N_SAMPLES)

    # --- Plot 1 data: single-node Pd vs SNR, simulation + theory ----------------
    pd_sim_1, pd_th_1 = [], []
    for g in snr_lin:
        lam = threshold(g)
        pd_sim_1.append(monte_carlo_probability(lam, g, N_SAMPLES, args.trials, True, rng))
        pd_th_1.append(theoretical_pd(lam, g, N_SAMPLES))
    # False-alarm check at the same thresholds (should sit near PFA_TARGET).
    pfa_sim_1 = [
        monte_carlo_probability(threshold(g), g, N_SAMPLES, args.trials, False, rng)
        for g in snr_lin
    ]
    out["single_node"] = {"pd_sim": pd_sim_1, "pd_theory": pd_th_1, "pfa_sim": pfa_sim_1}

    # --- Plot 3 data: cooperative OR rule, K = 1, 3, 5 -------------------------
    coop_sim, coop_th = {}, {}
    for k in NODES:
        coop_sim[k] = [
            monte_carlo_cooperative(threshold(g), g, N_SAMPLES, args.trials, k, True, rng)
            for g in snr_lin
        ]
        coop_th[k] = [cooperative_pd(p, k) for p in pd_th_1]
    coop_pfa_sim = {
        k: monte_carlo_cooperative(threshold(1.0), 1.0, N_SAMPLES, args.trials, k, False, rng)
        for k in NODES
    }
    out["cooperative"] = {"pd_sim": coop_sim, "pd_theory": coop_th, "pfa_sim": coop_pfa_sim}

    # --- Plot 2 data: ROC at -10 dB (pooled empirical sweep) -------------------
    # Common random numbers: one pooled H0 and one pooled H1 statistic set per K,
    # thresholds swept over the same data, so each empirical ROC is monotone.
    g_roc = snr_db_to_linear(ROC_SNR_DB)
    pfa_grid = np.linspace(0.001, 0.999, ROC_POINTS)
    roc = {}
    for k in NODES:
        s0 = cooperative_statistics(g_roc, N_SAMPLES, args.trials, k, False, rng)
        s1 = cooperative_statistics(g_roc, N_SAMPLES, args.trials, k, True, rng)
        lams = np.quantile(s0, 1.0 - pfa_grid)
        roc[k] = {"pfa": [float(np.mean(s0 > l)) for l in lams],
                  "pd": [float(np.mean(s1 > l)) for l in lams]}
    out["roc_at_minus10db"] = roc

    # --- Plots ------------------------------------------------------------------
    INK, TEAL, ORG, GR = "#14213d", "#0f9d8a", "#e07a1f", "#6b7280"
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False})

    def style(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    ax.plot(SNR_DB, pd_sim_1, "o", color=TEAL, label="Simulation (1 node)")
    ax.plot(SNR_DB, pd_th_1, "-", color=INK, label="Theory (Gaussian approx.)")
    ax.legend(frameon=False)
    style(ax, "SNR (dB)", "Probability of detection $P_d$")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "pd_vs_snr.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    for k, c, lab in ((1, GR, "Single node"), (3, TEAL, "3-node OR"), (5, ORG, "5-node OR")):
        ax.plot(roc[k]["pfa"], roc[k]["pd"], color=c, lw=2.2, label=lab)
    ax.plot([0, 1], [0, 1], ls=":", color="#9ca3af")
    ax.legend(frameon=False, loc="lower right")
    style(ax, "Probability of false alarm $P_{fa}$", "Probability of detection $P_d$")
    ax.set_title(f"ROC at SNR = {ROC_SNR_DB} dB")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "roc_minus10db.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    for k, c, lab in ((1, GR, "Single node"), (3, TEAL, "3 nodes (OR)"), (5, ORG, "5 nodes (OR)")):
        ax.plot(SNR_DB, coop_sim[k], "o-", color=c, ms=4, lw=1.8, label=lab)
        ax.plot(SNR_DB, coop_th[k], "--", color=c, lw=1.0, alpha=0.6)
    ax.legend(frameon=False)
    style(ax, "SNR (dB)", "Probability of detection $P_d$")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "cooperative_or.png"))
    plt.close(fig)

    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Console summary at the operating point used in the report.
    i10 = list(SNR_DB).index(-10.0)
    print(f"threshold normalization: sigma^2 = 1, signal power = SNR (linear)")
    print(f"Pfa target {PFA_TARGET}; simulated single-node Pfa ~ {np.mean(pfa_sim_1):.4f}")
    print(f"Pd at -10 dB: 1 node {pd_sim_1[i10]:.3f} | 3 nodes {coop_sim[3][i10]:.3f} | "
          f"5 nodes {coop_sim[5][i10]:.3f} (theory {pd_th_1[i10]:.3f} single)")
    print(f"OR-rule Pfa (simulated): 1 node {coop_pfa_sim[1]:.3f} | "
          f"3 nodes {coop_pfa_sim[3]:.3f} | 5 nodes {coop_pfa_sim[5]:.3f}")


if __name__ == "__main__":
    main()
