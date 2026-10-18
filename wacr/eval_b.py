"""Method B evaluation with a scalar or per-query tau.

Copied from Outputs/eval_wrcp_fdiv.py (_eval_b, _eval_b_per_query).
"""
from __future__ import annotations

import numpy as np


def _eval_b(wacr, job, tau: float):
    res = wacr.evaluate_adaptive(
        job["test_sims"], tau, job["phi_test"], job["true_test"], min_set_size=0
    )
    valid = job["true_test"] >= 0
    miss = (~res.contains_true[valid]).astype(np.float64)
    return float(res.r_hat), float(res.avg_set_size), miss, res.set_sizes[valid]


def _eval_b_per_query(job, tau_q: np.ndarray):
    phi = np.asarray(job["phi_test"], dtype=np.float64).reshape(-1)
    lim = tau_q * phi
    accepted = (1.0 - job["test_sims"]) <= lim[:, None]
    set_sizes = accepted.sum(axis=1).astype(np.float64)
    true = job["true_test"]
    valid = true >= 0
    contains = np.zeros(true.shape[0], dtype=bool)
    rows = np.flatnonzero(valid)
    contains[rows] = accepted[rows, true[rows]]
    miss = (~contains[valid]).astype(np.float64)
    sizes = set_sizes[valid]
    r = float(miss.mean()) if miss.size else float("nan")
    c = float(sizes.mean()) if sizes.size else float("nan")
    return r, c, miss, sizes


def empty_pct(sizes) -> float:
    sizes = np.asarray(sizes, dtype=np.float64).reshape(-1)
    n = int(sizes.size)
    if n == 0:
        return float("nan")
    return 100.0 * float(np.mean(sizes <= 0.0))
