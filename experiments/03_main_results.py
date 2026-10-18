#!/usr/bin/env python3
"""T1 main marginal results (alpha=0.10, Method B) and figures B / A_main."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baselines.split_cp import split_cp_threshold
from baselines.t1 import (
    fit_baseline_weights,
    run_condcp,
    run_dscp,
    run_ltt,
    run_paccs,
    run_qtc,
    run_rafr,
    run_wrcp,
)
from experiments._common import (
    copy_to_outputs,
    empty_pct,
    hist_w,
    load_jobs,
    out_dir,
    write_csv,
)
from wacr.boot import N_BOOT, _group_index, _seed_for, _strat_boot_means, paper_comb
from wacr.figures import (
    A_MAIN_ALL,
    T1_TO_A_MAIN,
    T1_TO_B_MID,
    WACR_G1,
    WACR_G15,
    plot_a_main,
    plot_b_main,
)
A10 = 0.10
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

T1_SPEC = [
    ("No weight correction", "Split CP", "textbook mass 1 at +inf"),
    ("No weight correction", "QTC", "min rule, Method B, conf=t"),
    ("No weight correction", "Conditional CP", "one-hot 20 bins on t"),
    ("No weight correction", "RA-FR", "percentile budget"),
    ("Weight correction", "WCP-KMM", "frozen"),
    ("Weight correction", "WCP-KLIEP", "frozen"),
    ("Weight correction", "WCP-DC", "frozen"),
    ("Weight correction", "PAC-CS", "PS-W AdaFace"),
    ("Weight correction", "WRCP", "XGB DINOv2 per-query, rho*"),
    ("Weight correction", "LTT-IW", "KMM QP WSR"),
    ("Weight correction", "DS-CP", "XGB DINOv2 gamma=1"),
    ("Proposed", "WACR", "Gamma=1"),
    ("Proposed", "WACR", "Gamma=1.5"),
    ("Proposed", "WACR-PAC", "Gamma=1, Bernstein"),
]


def strat_ci(miss, ids, *key) -> tuple[float, float]:
    groups = _group_index(np.asarray(ids))
    rng = np.random.default_rng(_seed_for(0, *key, N_BOOT))
    boot = _strat_boot_means(np.asarray(miss, dtype=np.float64), groups, rng, N_BOOT)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def wacr_g1_ci(miss, ids, cal: str, alpha: float) -> tuple[float, float]:
    groups = _group_index(np.asarray(ids))
    rng = np.random.default_rng(_seed_for(paper_comb(cal), "B", alpha, 1.0, N_BOOT))
    boot = _strat_boot_means(np.asarray(miss, dtype=np.float64), groups, rng, N_BOOT)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def eval_wcp(wacr, job, w, alpha: float):
    from wacr.protocol import _eval

    _tau, r, c, miss, sizes = _eval(wacr, job, job["s_g"], w, alpha, True)
    return r, c, empty_pct(sizes), miss


def eval_split(wacr, job, alpha: float):
    tau = split_cp_threshold(job["s_g"], alpha)
    res = wacr.evaluate_adaptive(
        job["test_sims"], tau, job["phi_test"], job["true_test"], min_set_size=0
    )
    valid = job["true_test"] >= 0
    sizes = res.set_sizes[valid]
    miss = (~res.contains_true[valid]).astype(np.float64)
    return float(res.r_hat), float(res.avg_set_size), empty_pct(sizes), miss


def eval_wacr(wacr, job, alpha: float, *, gamma: float = 1.0, bound: str = "conformal"):
    from wacr.protocol import W_MAX, _eval

    w_hist = hist_w(job)
    if bound == "conformal" and abs(float(gamma) - 1.0) > 1e-12:
        tau = wacr.robust_weighted_conformal_threshold(
            job["s_g"], w_hist, alpha=alpha, gamma=float(gamma), w_test=W_MAX
        )
        res = wacr.evaluate_adaptive(
            job["test_sims"], tau, job["phi_test"], job["true_test"], min_set_size=0
        )
        valid = job["true_test"] >= 0
        sizes = res.set_sizes[valid]
        miss = (~res.contains_true[valid]).astype(np.float64)
        return float(res.r_hat), float(res.avg_set_size), empty_pct(sizes), miss
    _t, r, c, miss, sizes = _eval(
        wacr, job, job["s_g"], w_hist, alpha, True, bound=bound
    )
    return r, c, empty_pct(sizes), miss


def method_at_alpha(wacr, job, method: str, variant: str, alpha: float):
    if method == "Split CP":
        return eval_split(wacr, job, alpha)
    if method == "QTC":
        return run_qtc(job, alpha)
    if method == "Conditional CP":
        return run_condcp(job, alpha)
    if method == "RA-FR":
        return run_rafr(job, alpha)
    if method == "WCP-KMM":
        return eval_wcp(wacr, job, job["w_kmm"], alpha)
    if method == "WCP-KLIEP":
        return eval_wcp(wacr, job, job["w_kliep"], alpha)
    if method == "WCP-DC":
        return eval_wcp(wacr, job, job["w_dc"], alpha)
    if method == "PAC-CS":
        return run_paccs(wacr, job, alpha)
    if method == "WRCP":
        return run_wrcp(wacr, job, alpha)
    if method == "LTT-IW":
        return run_ltt(wacr, job, alpha)
    if method == "DS-CP":
        return run_dscp(wacr, job, alpha)
    if method == "WACR" and variant == "Gamma=1":
        return eval_wacr(wacr, job, alpha, gamma=1.0)
    if method == "WACR" and variant == "Gamma=1.5":
        return eval_wacr(wacr, job, alpha, gamma=1.5)
    if method == "WACR-PAC":
        return eval_wacr(wacr, job, alpha, bound="bernstein")
    raise KeyError(method, variant)


def main() -> None:
    wacr, jobs = load_jobs()
    fit_baseline_weights(wacr, jobs)

    sweep = []
    t1 = {}
    forest: dict = {}
    for job in jobs:
        cal = job["cal"]
        ids = job["test_ids"]
        print(f"=== T1 / A_main grid {cal} ===", flush=True)
        for group, method, variant in T1_SPEC:
            alphas = ALPHAS if T1_TO_A_MAIN.get((method, variant)) else [A10]
            for a in alphas:
                r, c, emp, miss = method_at_alpha(wacr, job, method, variant, a)
                am_name = T1_TO_A_MAIN.get((method, variant))
                rec = {
                    "cal": cal,
                    "method": am_name or method,
                    "variant": variant,
                    "alpha": a,
                    "r_hat": r,
                    "set_size": c,
                    "empty_pct": emp,
                    "r_lo": "",
                    "r_hi": "",
                }
                if method == "WACR" and variant == "Gamma=1":
                    lo, hi = wacr_g1_ci(miss, ids, cal, a)
                    rec["r_lo"] = lo
                    rec["r_hi"] = hi
                if am_name in A_MAIN_ALL:
                    sweep.append(rec)
                if abs(a - A10) < 1e-12:
                    if method == "WACR":
                        t1[(cal, "WACR", variant)] = (r, c, emp)
                    elif method == "WACR-PAC":
                        t1[(cal, "WACR-PAC")] = (r, c, emp)
                    else:
                        t1[(cal, method)] = (r, c, emp)
                    if method == "WACR" and variant == "Gamma=1":
                        mid = "WACR_1"
                    elif method == "WACR" and variant == "Gamma=1.5":
                        mid = "WACR_15"
                    elif method == "WACR-PAC":
                        mid = "WACR_PAC"
                    else:
                        mid = T1_TO_B_MID.get(method)
                    if mid:
                        lo, hi = strat_ci(miss, ids, "figB", cal, method, variant, A10)
                        forest.setdefault(mid, {})[cal] = {
                            "r": r,
                            "c": c,
                            "lo": lo,
                            "hi": hi,
                        }
                    print(f"  {method} {variant} a=0.10 R={r:.6f} |C|={c:.4f}", flush=True)

    def cell(cal, method, variant):
        if method == "WACR":
            return t1[(cal, "WACR", variant)]
        if method == "WACR-PAC":
            return t1[(cal, "WACR-PAC")]
        return t1[(cal, method)]

    rows = []
    for group, method, variant in T1_SPEC:
        tf = cell("TinyFace", method, variant)
        xq = cell("XQLFW", method, variant)
        rows.append(
            {
                "group": group,
                "method": method,
                "variant": variant,
                "labels": "no",
                "TF_R": tf[0],
                "TF_C": tf[1],
                "TF_empty": tf[2],
                "XQ_R": xq[0],
                "XQ_C": xq[1],
                "XQ_empty": xq[2],
            }
        )
        print(f"T1 {method} {variant} TF_R={tf[0]} XQ_R={xq[0]}", flush=True)

    write_csv(out_dir() / "t1_marginal.csv", rows)
    sweep_out = [
        {
            "cal": r["cal"],
            "method": r["method"],
            "alpha": r["alpha"],
            "r_hat": r["r_hat"],
            "set_size": r["set_size"],
            "r_lo": r["r_lo"],
            "r_hi": r["r_hi"],
        }
        for r in sweep
        if r["method"] in A_MAIN_ALL
    ]
    write_csv(out_dir() / "a_main_alpha_sweep.csv", sweep_out)

    fig_dir = REPO / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    b_path = fig_dir / "B.png"
    a_path = fig_dir / "A_main.png"
    plot_b_main(forest, b_path)
    plot_a_main(sweep_out, a_path)
    copy_to_outputs(b_path, "B.png")
    copy_to_outputs(a_path, "A_main.png")


if __name__ == "__main__":
    main()
