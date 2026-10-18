#!/usr/bin/env python3
"""Table 9 one-at-a-time hyperparameter sensitivity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments._common import load_jobs, out_dir, write_csv
from wacr.protocol import BETA, W_MAX, _hist_ratio_w
from wacr.weights import finalize_wcp_weights

GAMMAS = np.unique(
    np.round(
        np.concatenate(
            [
                np.array([1.0]),
                np.linspace(1.05, 2.0, 20),
                np.logspace(np.log10(2.0), np.log10(8.0), 16),
            ]
        ),
        4,
    )
)


def _hist_w(t_cal, t_te, n_bins=20, w_max=W_MAX):
    raw = _hist_ratio_w(t_cal, t_te, n_bins=n_bins)
    return finalize_wcp_weights(raw, int(t_cal.shape[0]), w_max=w_max, epsilon=0.1)


def gamma_hat_from_curve(gammas, rhats, alpha) -> float:
    for g, r in zip(gammas, rhats):
        if r <= alpha + 1e-12:
            return float(g)
    return float("inf")


def gamma_vac(alpha, n, bcap) -> float:
    return float(np.sqrt(alpha * n / ((1.0 - alpha) * bcap)))


def main() -> None:
    wacr, jobs = load_jobs()
    settings = [("M", m, W_MAX, BETA) for m in (10, 20, 40)]
    settings += [("B", 20, b, BETA) for b in (10, 30, 100)]
    settings += [("beta", 20, W_MAX, b) for b in (0.25, 0.5, 1.0)]
    a = 0.10
    rows = []
    for job in jobs:
        for vary, m_bins, bcap, beta in settings:
            w = _hist_w(job["t_cal"], job["t_te"], n_bins=m_bins, w_max=bcap)
            if abs(beta - BETA) > 1e-12:
                phi = np.clip(1.0 - job["t_cal"], 0.0, 1.0) + beta
                scores = job["s_base"] / phi
                phi_te = np.clip(1.0 - job["t_te"], 0.0, 1.0) + beta
            else:
                scores = job["s_g"]
                phi_te = job["phi_test"]
            rhats, cs = [], []
            r1 = c1 = float("nan")
            for g in GAMMAS:
                tau = wacr.robust_weighted_conformal_threshold(
                    scores, w, alpha=a, gamma=float(g), w_test=bcap
                )
                res = wacr.evaluate_adaptive(
                    job["test_sims"], tau, phi_te, job["true_test"], min_set_size=0
                )
                r, c = float(res.r_hat), float(res.avg_set_size)
                rhats.append(r)
                cs.append(c)
                if abs(g - 1.0) < 1e-12:
                    r1, c1 = r, c
            ghat = gamma_hat_from_curve(GAMMAS, rhats, a)
            c_at = cs[int(np.argmin(np.abs(GAMMAS - ghat)))] if np.isfinite(ghat) else float("nan")
            r_at = rhats[int(np.argmin(np.abs(GAMMAS - ghat)))] if np.isfinite(ghat) else float("nan")
            n = job["n_calib"]
            rows.append(
                {
                    "combination": job["comb"],
                    "cal": job["cal"],
                    "vary": vary,
                    "M": m_bins,
                    "B": bcap,
                    "beta": beta,
                    "alpha": a,
                    "r_hat_gamma1": r1,
                    "gamma_hat": ghat,
                    "r_hat_at_hat": r_at,
                    "set_size_at_hat": c_at,
                    "gamma_vac": gamma_vac(a, n, bcap),
                    "set_size_gamma1": c1,
                }
            )
            print(
                f"  {job['cal']} {vary} M={m_bins} B={bcap} b={beta} R(1)={r1:.4f} Ghat={ghat}",
                flush=True,
            )
    write_csv(out_dir() / "table9_sensitivity.csv", rows)


if __name__ == "__main__":
    main()
