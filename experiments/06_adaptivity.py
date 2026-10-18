#!/usr/bin/env python3
"""Fig 6 per-query set size vs true-match rank.

Copied from Outputs/make_fig_c5_adaptivity.py (run_panel + write_csv).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments._common import emit_figure, hist_w, load_jobs, out_dir, read_csv
from wacr.figures import plot_adaptivity
from wacr.identity import SCFACE_DISTANCES, scface_distance_of
from wacr.protocol import W_MAX

ALPHA = 0.10
N_PROBE = 975
DIST_M = {1: 4.20, 2: 2.60, 3: 1.00}
DIST_ORDER = (1, 2, 3)
PANELS = (
    {"key": "a", "cal": "TinyFace", "gamma": 1.0},
    {"key": "b", "cal": "XQLFW", "gamma": 1.0},
    {"key": "c", "cal": "XQLFW", "gamma": 1.4},
)


def verify_distance_map(paths: list[str]) -> np.ndarray:
    codes = np.array([scface_distance_of(p) for p in paths], dtype=np.int64)
    if not set(codes.tolist()).issubset(set(SCFACE_DISTANCES)):
        raise SystemExit("distance codes outside {1,2,3}")
    return codes


def main() -> None:
    wacr, jobs = load_jobs()
    by_cal = {j["cal"]: j for j in jobs}
    paths = list(jobs[0]["test_paths"])
    dist = verify_distance_map(paths)
    rows = []
    for spec in PANELS:
        job = by_cal[spec["cal"]]
        true = np.asarray(job["true_test"])
        valid = true >= 0
        if int(valid.sum()) != N_PROBE or int(true.size) != N_PROBE:
            raise SystemExit(f"{spec['key']}: n_eval mismatch")
        if not np.all(valid):
            raise SystemExit(f"{spec['key']}: some probes have no gallery match")
        t = np.asarray(job["test_sims"].max(axis=1), dtype=np.float64)
        w = hist_w(job)
        tau = wacr.robust_weighted_conformal_threshold(
            job["s_g"], w, alpha=ALPHA, gamma=float(spec["gamma"]), w_test=W_MAX
        )
        res = wacr.evaluate_adaptive(
            job["test_sims"], tau, job["phi_test"], true, min_set_size=0
        )
        size = res.set_sizes.astype(np.int64)
        hit = res.contains_true.astype(bool)
        r_hat = float((~hit).mean())
        mean_c = float(size.mean())
        print(
            f"panel {spec['key']}: Gamma={spec['gamma']} R-hat={r_hat:.6f} "
            f"mean |C|={mean_c:.4f} tau={tau:.6f}",
            flush=True,
        )
        pair = f"{spec['cal']} to SCFace"
        for i in range(N_PROBE):
            rows.append(
                {
                    "panel": spec["key"],
                    "pair": pair,
                    "gamma": f"{spec['gamma']:.1f}",
                    "probe": Path(paths[i]).name,
                    "t": f"{t[i]:.10g}",
                    "tau": f"{tau:.10g}",
                    "set_size": int(size[i]),
                    "hit": int(hit[i]),
                    "distance_code": int(dist[i]),
                    "distance_m": f"{DIST_M[int(dist[i])]:.2f}",
                }
            )
    path = out_dir() / "fig6_adaptivity.csv"
    fields = [
        "panel",
        "pair",
        "gamma",
        "probe",
        "t",
        "tau",
        "set_size",
        "hit",
        "distance_code",
        "distance_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"Wrote {path}", flush=True)
    emit_figure(plot_adaptivity, read_csv(path), "fig_adaptivity")


if __name__ == "__main__":
    main()
