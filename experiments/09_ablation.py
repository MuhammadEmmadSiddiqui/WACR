#!/usr/bin/env python3
"""Table 8 ablation (configs a-g).

Copied from Outputs/ablation/run_ablation.py run_results / write_results_csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments._common import load_jobs, out_dir, write_csv
from wacr.boot import GAMMAS
from wacr.protocol import BETA, W_MAX
from wacr.weights import finalize_wcp_weights
from wacr.protocol import _hist_ratio_w

ALPHA = 0.10


def hist_w_pair(t_cal: np.ndarray, t_te: np.ndarray) -> np.ndarray:
    return finalize_wcp_weights(
        _hist_ratio_w(t_cal, t_te, n_bins=20), int(t_cal.shape[0]), w_max=W_MAX, epsilon=0.1
    )


def _t_full(job) -> np.ndarray:
    return np.asarray(job["top1_full"], dtype=np.float64).reshape(-1)


def eval_pair(wacr, job, scores, w, gamma: float, adaptive: bool) -> tuple[float, float, float]:
    tau = wacr.robust_weighted_conformal_threshold(
        scores, w, alpha=ALPHA, gamma=float(gamma), w_test=W_MAX
    )
    if adaptive:
        res = wacr.evaluate_adaptive(
            job["test_sims"], tau, job["phi_test"], job["true_test"], min_set_size=0
        )
    else:
        res = wacr.evaluate_baseline(job["test_sims"], tau, job["true_test"], min_set_size=0)
    return float(tau), float(res.r_hat), float(res.avg_set_size)


def gstar(wacr, job, scores, w, adaptive: bool):
    ghat = float("inf")
    r_at = c_at = float("nan")
    r1 = c1 = float("nan")
    for g in GAMMAS:
        _t, r, c = eval_pair(wacr, job, scores, w, g, adaptive)
        if abs(g - 1.0) < 1e-12:
            r1, c1 = r, c
        if r <= ALPHA + 1e-12 and not np.isfinite(ghat):
            ghat, r_at, c_at = float(g), r, c
    if not np.isfinite(ghat):
        ghat, r_at, c_at = float("inf"), r1, c1
    return r1, c1, ghat, r_at, c_at


def _csv_row(pair, key, d):
    return {
        "pair": pair,
        "config": key,
        "R_hat": d.get("r", d.get("r1", "")),
        "mean_C": d.get("c", d.get("c1", "")),
        "R_hat_G1": d.get("r1", ""),
        "mean_C_G1": d.get("c1", ""),
        "gamma": d.get("g", ""),
        "phi_gallery": d.get("phi_gallery", ""),
    }


def main() -> None:
    wacr, jobs = load_jobs()
    rows = []
    for job in jobs:
        pair = f"{job['cal']}->SCFace"
        print(f"=== ablation {pair} ===", flush=True)
        t_m = job["t_cal"]
        t_f = _t_full(job)
        s_m = job["s_g"]
        s_a = job["s_base"]
        phi_f = np.clip(1.0 - t_f, 0.0, 1.0) + BETA
        s_f = s_a / phi_f
        w_u = np.ones(job["n_calib"], dtype=np.float64)
        w_hm = hist_w_pair(t_m, job["t_te"])
        w_hf = hist_w_pair(t_f, job["t_te"])
        rec = {}

        ta, ra, ca = eval_pair(wacr, job, s_a, w_u, 1.0, False)
        rec["a"] = {"r": ra, "c": ca, "g": 1.0, "tau": ta, "phi_gallery": "n/a (Method A)"}
        print(f"  (a) unw A G=1  R={ra:.6f} |C|={ca:.4f}", flush=True)

        tb, rb, cb = eval_pair(wacr, job, s_m, w_u, 1.0, True)
        rec["b"] = {
            "r": rb,
            "c": cb,
            "g": 1.0,
            "tau": tb,
            "phi_gallery": "k=65 identity-matched (s_ad_matched / top1_matched)",
        }
        print(f"  (b) unw B G=1  R={rb:.6f} |C|={cb:.4f}", flush=True)

        r1, c1, gh, r_at, c_at = gstar(wacr, job, s_f, w_hf, True)
        rec["c"] = {
            "r1": r1,
            "c1": c1,
            "g": gh,
            "r": r_at,
            "c": c_at,
            "phi_gallery": "full source gallery",
        }
        print(
            f"  (c) hist unmatched B  G=1 R={r1:.6f} |C|={c1:.4f}  G*={gh:.4f} "
            f"R={r_at:.6f} |C|={c_at:.4f}",
            flush=True,
        )

        r1a, c1a, gha, rat, cat = gstar(wacr, job, s_a, w_hm, False)
        rec["d"] = {"r1": r1a, "c1": c1a, "g": gha, "r": rat, "c": cat}
        print(f"  (d) hist matched A  G*={gha:.4f} R={rat:.6f} |C|={cat:.4f}", flush=True)

        te, re, ce = eval_pair(wacr, job, s_m, w_hm, 1.0, True)
        rec["e"] = {"r": re, "c": ce, "g": 1.0, "tau": te}
        print(f"  (e) hist matched B G=1  R={re:.6f} |C|={ce:.4f}", flush=True)

        r1e, c1e, ghf, rf, cf = gstar(wacr, job, s_m, w_hm, True)
        rec["f"] = {"r1": r1e, "c1": c1e, "g": ghf, "r": rf, "c": cf}
        print(f"  (f) hist matched B G*={ghf:.4f} R={rf:.6f} |C|={cf:.4f}", flush=True)

        tg, rg, cg = eval_pair(wacr, job, s_m, w_hm, 1.5, True)
        rec["g"] = {"r": rg, "c": cg, "g": 1.5, "tau": tg}
        print(f"  (g) hist matched B G=1.5  R={rg:.6f} |C|={cg:.4f}", flush=True)

        for key in "abcdefg":
            rows.append(_csv_row(pair, key, rec[key]))

    write_csv(out_dir() / "table8_ablation.csv", rows)


if __name__ == "__main__":
    main()
