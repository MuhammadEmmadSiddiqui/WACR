#!/usr/bin/env python3
"""Table 5 rank budgets vs WACR-PAC."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments._common import hist_w, load_jobs, out_dir, write_csv
from wacr.protocol import ALPHAS, BETA, DELTA
from wacr.rafr import k_cut, matched_rank_and_t, test_ranks

DELTA_KAPPA = 0.5
SEED = 0


def calibrate_kappa(ranks, n_gal, phi, w_tilde, alpha, n_te, mode: str):
    n = int(ranks.size)
    w_max = float(w_tilde.max())
    kappa_max = float(n_te)
    grid_size = int(math.ceil((kappa_max - 1.0) / DELTA_KAPPA)) + 1
    log_term = math.log(2.0 * grid_size / DELTA)
    var_coef = 2.0 * log_term / n
    bias = 7.0 * w_max * log_term / (3.0 * (n - 1))
    kappa_hat = None
    mu_at = u_at = float("nan")
    for k in range(grid_size):
        kappa = 1.0 + k * DELTA_KAPPA
        k_cal, _ = k_cut(kappa, phi, n_gal, n_te, mode)
        losses = (ranks > k_cal).astype(np.float64)
        wl = w_tilde * losses
        mu = float(wl.mean())
        v = max(0.0, float(np.var(wl, ddof=1)))
        u = mu + math.sqrt(var_coef * v) + bias
        if u <= alpha + 1e-12:
            kappa_hat = float(kappa)
            mu_at, u_at = mu, u
            break
    return kappa_hat, mu_at, u_at, bias, w_max, grid_size


def table5() -> None:
    wacr, jobs = load_jobs()
    rows = []
    for job in jobs:
        t, rank, n_gal = matched_rank_and_t(
            job["calib_sims"],
            job["calib_ids"],
            job["cg_ids"],
            n_gallery_ids=int(job["n_ids_test"]),
            seed=SEED,
        )
        err = float(np.max(np.abs(t - job["t_cal"])))
        if err > 1e-5:
            raise RuntimeError("matched t does not match gmatch cache")
        phi_cal = np.clip(1.0 - t, 0.0, 1.0) + BETA
        phi_te = np.clip(1.0 - job["t_te"], 0.0, 1.0) + BETA
        te_rank = test_ranks(job["test_sims"], job["true_test"])
        valid = te_rank > 0
        n_te_gal = int(job["test_sims"].shape[1])
        w_hist = hist_w(job)
        setups = [
            ("Unweighted", np.ones(job["n_calib"], dtype=np.float64)),
            ("WACR (histogram)", w_hist),
        ]
        for mode in ("count", "percentile"):
            for wname, w_raw in setups:
                w_sum = float(w_raw.sum())
                w_tilde = (job["n_calib"] / w_sum) * w_raw
                for a in ALPHAS:
                    khat, mu, u, bias, wmax, gsz = calibrate_kappa(
                        rank, n_gal, phi_cal, w_tilde, a, n_te_gal, mode
                    )
                    rec = {
                        "cal": job["cal"],
                        "weight": wname,
                        "loss": mode,
                        "alpha": a,
                        "kappa_hat": "",
                        "mean_K": "",
                        "r_hat": "",
                        "controlled": 0,
                        "status": "infeasible",
                        "bias": bias,
                        "w_tilde_max": wmax,
                        "mean_n_gal": float(n_gal.mean()),
                        "n_te_gal": n_te_gal,
                    }
                    if khat is not None:
                        _, k_final = k_cut(
                            khat, phi_te, np.full(phi_te.shape, n_te_gal), n_te_gal, mode
                        )
                        miss = (te_rank[valid] > k_final[valid]).astype(np.float64)
                        r = float(miss.mean())
                        mk = float(k_final[valid].mean())
                        rec.update(
                            {
                                "kappa_hat": khat,
                                "mean_K": mk,
                                "r_hat": r,
                                "controlled": int(r <= a + 1e-12),
                                "status": "ok",
                            }
                        )
                    rows.append(rec)
                    print(
                        f"  {job['cal']} {mode} {wname} a={a:.2f} {rec['status']}",
                        flush=True,
                    )
    write_csv(out_dir() / "table5_rank_budget.csv", rows)


def main() -> None:
    table5()


if __name__ == "__main__":
    main()
