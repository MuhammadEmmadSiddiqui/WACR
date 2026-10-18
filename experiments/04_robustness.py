#!/usr/bin/env python3
"""Table 3 robustness and Fig 4 Gamma sensitivity.

Runs the Jin Gamma sweep and identity-stratified bootstrap (n_boot=1000)
from eval_sensitivity_gamma.py / make_full_alpha_gamma_table.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments._common import emit_figure, hist_w, load_jobs, out_dir, read_csv, write_csv
from wacr.figures import plot_sensitivity
from wacr.boot import (
    ALPHAS_FULL,
    FROZEN_ALPHAS,
    GAMMAS,
    HEADLINE_ALPHA,
    N_BOOT,
    _cluster_boot_means,
    _first_cross,
    _group_index,
    _pct,
    _seed_for,
    _strat_boot_means,
    _tau_cell,
    gamma_vac,
    pair_name,
    paper_comb,
)
from wacr.protocol import W_MAX

G_LO, G_HI = 1.0, 5.0
N_GAL = 585
N_CAL = {"TinyFace": 2292, "XQLFW": 2296}
B_CAP = 30.0
FIG4_FIELDS = [
    "kind",
    "combination",
    "cal",
    "method",
    "alpha",
    "gamma",
    "r_hat",
    "strat_lo",
    "strat_hi",
    "set_size",
    "gamma_hat",
    "gamma_cons",
    "gamma_vac",
    "n",
    "n_gallery",
    "note",
]


def _f(x) -> float:
    if x in ("", None, "inf", "nan"):
        return float("nan")
    return float(x)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) < tol


def _eval(wacr, job: dict, tau: float, adaptive: bool):
    if adaptive:
        res = wacr.evaluate_adaptive(
            job["test_sims"], tau, job["phi_test"], job["true_test"], min_set_size=0
        )
    else:
        res = wacr.evaluate_baseline(job["test_sims"], tau, job["true_test"], min_set_size=0)
    valid = job["true_test"] >= 0
    miss = (~res.contains_true[valid]).astype(np.float64)
    return float(res.r_hat), float(res.avg_set_size), miss


def _run_job(wacr, job: dict, *, n_boot: int) -> list[dict]:
    rows: list[dict] = []
    w = job["w"]
    groups = _group_index(job["test_ids"])
    comb = job["comb"]
    print(
        f"\n=== {job['cal']} -> {job['tst']}  hist n={job['n_calib']} "
        f"n_eval={job['n_eval']}  |G|={job['n_gal']}  max w={float(w.max()):.2f} ===",
        flush=True,
    )
    for meth, scores, adaptive in (
        ("A", job["s_base"], False),
        ("B", job["s_g"], True),
    ):
        tau1 = wacr.weighted_conformal_threshold(scores, w, alpha=HEADLINE_ALPHA, w_test=W_MAX)
        tau1r = wacr.robust_weighted_conformal_threshold(
            scores, w, alpha=HEADLINE_ALPHA, gamma=1.0, w_test=W_MAX
        )
        same_inf = (not np.isfinite(tau1)) and (not np.isfinite(tau1r))
        same_fin = np.isfinite(tau1) and np.isfinite(tau1r) and abs(tau1 - tau1r) <= 1e-9
        if not (same_inf or same_fin):
            raise SystemExit(f"Gamma=1 mismatch {job['cal']} {meth}: WCP {tau1} vs Jin {tau1r}")
        for a in ALPHAS_FULL:
            for g in GAMMAS:
                tau = wacr.robust_weighted_conformal_threshold(
                    scores, w, alpha=a, gamma=float(g), w_test=W_MAX
                )
                r, c, miss = _eval(wacr, job, tau, adaptive)
                rec = {
                    "combination": pair_name(job["cal"]),
                    "cal": job["cal"],
                    "test": job["tst"],
                    "weight": "hist ratio",
                    "method": meth,
                    "alpha": a,
                    "gamma": float(g),
                    "tau": _tau_cell(tau),
                    "tau_inf": int(not np.isfinite(tau)),
                    "r_hat": r,
                    "set_size": c,
                    "n_gallery": job["n_gal"],
                    "full_gallery": int(c + 1e-9 >= job["n_gal"]),
                    "controlled": int(r <= a + 1e-12),
                    "strat_lo": "",
                    "strat_hi": "",
                    "cluster_lo": "",
                    "cluster_hi": "",
                    "n_boot": n_boot,
                    "seed_comb": comb,
                }
                rng = np.random.default_rng(_seed_for(comb, meth, a, float(g), n_boot))
                strat = _strat_boot_means(miss, groups, rng, n_boot)
                clus = _cluster_boot_means(miss, groups, rng, n_boot)
                rec["strat_lo"] = _pct(strat, 0.025)
                rec["strat_hi"] = _pct(strat, 0.975)
                rec["cluster_lo"] = _pct(clus, 0.025)
                rec["cluster_hi"] = _pct(clus, 0.975)
                rows.append(rec)
            ghat = _first_cross(
                np.array(
                    [
                        x["gamma"]
                        for x in rows
                        if x["method"] == meth and abs(x["alpha"] - a) < 1e-12 and x["cal"] == job["cal"]
                    ]
                ),
                np.array(
                    [
                        x["r_hat"]
                        for x in rows
                        if x["method"] == meth and abs(x["alpha"] - a) < 1e-12 and x["cal"] == job["cal"]
                    ]
                ),
                a,
            )
            print(f"  Method {meth} a={a:.2f}  Gamma-hat={ghat}", flush=True)
    return rows


def _ctrl_flags(r_hat: float, lo_d: float, hi_d: float, alpha: float) -> dict:
    point = int(r_hat <= alpha + 1e-12)
    covers = int(lo_d <= 0.0 + 1e-12 and hi_d >= 0.0 - 1e-12)
    above = int(lo_d > 0.0 + 1e-12)
    below = int(hi_d < 0.0 - 1e-12)
    if point and covers:
        flag = "yes / covers 0"
    elif point and below:
        flag = "yes / below 0"
    elif point and above:
        flag = "yes / above 0"
    elif (not point) and covers:
        flag = "no / covers 0"
    elif (not point) and above:
        flag = "no / above 0"
    else:
        flag = "no / below 0"
    return {
        "controlled_point": point,
        "ci_covers_0": covers,
        "ci_entirely_above_0": above,
        "ci_entirely_below_0": below,
        "controlled_flag": flag,
    }


def _sub_gamma(sweep: list[dict], cal: str, a: float, meth: str = "B") -> list[dict]:
    sub = [
        r
        for r in sweep
        if r["cal"] == cal and r["method"] == meth and _close(_f(r["alpha"]), a)
    ]
    return sorted(sub, key=lambda r: _f(r["gamma"]))


def _table_row(sweep: list[dict], cal: str, tst: str, a: float, source: str) -> dict:
    sub = _sub_gamma(sweep, cal, a)
    if not sub:
        raise RuntimeError(f"no sweep rows for {cal} a={a}")
    g = np.array([_f(r["gamma"]) for r in sub], dtype=np.float64)
    rh = np.array([_f(r["r_hat"]) for r in sub], dtype=np.float64)
    slo = np.array([_f(r["strat_lo"]) for r in sub], dtype=np.float64)
    shi = np.array([_f(r["strat_hi"]) for r in sub], dtype=np.float64)
    ghat = _first_cross(g, rh, a)
    gopt = _first_cross(g, slo, a)
    gcons = _first_cross(g, shi, a)
    at1 = next(r for r in sub if _close(_f(r["gamma"]), 1.0))
    atg = next((r for r in sub if _close(_f(r["gamma"]), ghat)), None) if np.isfinite(ghat) else None
    r1 = _f(at1["r_hat"])
    lo_d = _f(at1["strat_lo"]) - a
    hi_d = _f(at1["strat_hi"]) - a
    flags = _ctrl_flags(r1, lo_d, hi_d, a)
    return {
        "row_type": "alpha",
        "combination": pair_name(cal),
        "cal": cal,
        "test": tst,
        "method": "B",
        "weight": "hist ratio",
        "alpha": a,
        "r_hat_gamma1": r1,
        "ci_lo_r_minus_alpha": lo_d,
        "ci_hi_r_minus_alpha": hi_d,
        "n_boot": int(_f(at1.get("n_boot", N_BOOT))),
        **flags,
        "set_size_gamma1": _f(at1["set_size"]),
        "gamma_hat": ghat if np.isfinite(ghat) else float("inf"),
        "gamma_opt_strat": gopt if np.isfinite(gopt) else float("inf"),
        "gamma_cons_strat": gcons if np.isfinite(gcons) else float("inf"),
        "r_hat_at_hat": _f(atg["r_hat"]) if atg else "",
        "set_size_at_hat": _f(atg["set_size"]) if atg else "",
        "source": source,
        "attained_at_alpha": "",
        "rhat_le_alpha": "yes" if flags["controlled_point"] else "no",
        "ci_on_r_minus_alpha": (
            "below 0"
            if flags["ci_entirely_below_0"]
            else ("above 0" if flags["ci_entirely_above_0"] else "covers 0")
        ),
    }


def _shift_row(alpha_rows: list[dict], cal: str, tst: str, *, le20: bool) -> dict:
    sub = [r for r in alpha_rows if r["cal"] == cal]
    if le20:
        sub = [r for r in sub if _f(r["alpha"]) <= 0.20 + 1e-12]
        label = "shift_le20"
        a_label = "max, alpha<=0.20"
    else:
        label = "shift_full"
        a_label = "max, full grid"
    gstar = max(_f(r["gamma_hat"]) for r in sub)
    at = max(sub, key=lambda r: (_f(r["gamma_hat"]), _f(r["alpha"])))
    return {
        "row_type": label,
        "combination": pair_name(cal),
        "cal": cal,
        "test": tst,
        "method": "B",
        "weight": "hist ratio",
        "alpha": a_label,
        "r_hat_gamma1": "",
        "ci_lo_r_minus_alpha": "",
        "ci_hi_r_minus_alpha": "",
        "n_boot": "",
        "controlled_point": "",
        "ci_covers_0": "",
        "ci_entirely_above_0": "",
        "ci_entirely_below_0": "",
        "controlled_flag": "",
        "set_size_gamma1": "",
        "gamma_hat": gstar,
        "gamma_opt_strat": "",
        "gamma_cons_strat": "",
        "r_hat_at_hat": "",
        "set_size_at_hat": "",
        "source": "derived",
        "attained_at_alpha": _f(at["alpha"]),
        "rhat_le_alpha": "--",
        "ci_on_r_minus_alpha": "--",
    }


def _empty_fig4(**kw) -> dict:
    rec = {k: "" for k in FIG4_FIELDS}
    rec.update(kw)
    return rec


def fig4_rows(sweep: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for rec in sweep:
        g = _f(rec["gamma"])
        if not (G_LO - 1e-12 <= g <= G_HI + 0.02):
            continue
        a = _f(rec["alpha"])
        if not any(_close(a, fa) for fa in FROZEN_ALPHAS):
            continue
        rows.append(
            _empty_fig4(
                kind="curve",
                combination=pair_name(rec["cal"]),
                cal=rec["cal"],
                method=rec["method"],
                alpha=f"{a:.2f}",
                gamma=f"{g:.10g}",
                r_hat=f"{_f(rec['r_hat']):.10g}",
                strat_lo="" if rec.get("strat_lo") in ("", None) else f"{_f(rec['strat_lo']):.10g}",
                strat_hi="" if rec.get("strat_hi") in ("", None) else f"{_f(rec['strat_hi']):.10g}",
                set_size=f"{_f(rec['set_size']):.10g}",
                n=N_CAL.get(rec["cal"], ""),
                n_gallery=int(_f(rec.get("n_gallery", N_GAL))),
                note="from frozen sensitivity_gamma.csv",
            )
        )
    for cal in ("TinyFace", "XQLFW"):
        n = N_CAL[cal]
        for meth in ("A", "B"):
            for a in FROZEN_ALPHAS:
                sub = _sub_gamma(sweep, cal, a, meth)
                g = np.array([_f(r["gamma"]) for r in sub], dtype=np.float64)
                rh = np.array([_f(r["r_hat"]) for r in sub], dtype=np.float64)
                shi = np.array([_f(r["strat_hi"]) for r in sub], dtype=np.float64)
                ghat = _first_cross(g, rh, a)
                gcons = _first_cross(g, shi, a)
                at1 = next(r for r in sub if _close(_f(r["gamma"]), 1.0))
                rows.append(
                    _empty_fig4(
                        kind="value",
                        combination=pair_name(cal),
                        cal=cal,
                        method=meth,
                        alpha=f"{a:.2f}",
                        r_hat=f"{_f(at1['r_hat']):.10g}",
                        set_size=f"{_f(at1['set_size']):.10g}",
                        gamma_hat=f"{ghat:.10g}" if np.isfinite(ghat) else "inf",
                        gamma_cons=f"{gcons:.10g}" if np.isfinite(gcons) else "inf",
                        gamma_vac=f"{gamma_vac(a, n, B_CAP):.10g}",
                        n=n,
                        n_gallery=N_GAL,
                        note="from frozen sensitivity_gamma_value.csv",
                    )
                )
    for cal, n in N_CAL.items():
        for a in FROZEN_ALPHAS:
            rows.append(
                _empty_fig4(
                    kind="vac",
                    combination=pair_name(cal),
                    cal=cal,
                    alpha=f"{a:.2f}",
                    gamma_vac=f"{gamma_vac(a, n, B_CAP):.10g}",
                    n=n,
                    n_gallery=N_GAL,
                    note="closed form sqrt(alpha*n/((1-alpha)*B)), B=30",
                )
            )
    return rows


def main() -> None:
    wacr, jobs = load_jobs()
    sweep = []
    for job in jobs:
        packed = {
            "comb": paper_comb(job["cal"]),
            "cal": job["cal"],
            "tst": job["tst"],
            "s_base": job["s_base"],
            "s_g": job["s_g"],
            "w": hist_w(job),
            "test_ids": job["test_ids"],
            "test_sims": job["test_sims"],
            "phi_test": job["phi_test"],
            "true_test": job["true_test"],
            "n_calib": job["n_calib"],
            "n_eval": job["n_eval"],
            "n_gal": int(job["test_sims"].shape[1]),
        }
        sweep.extend(_run_job(wacr, packed, n_boot=N_BOOT))

    table_rows = []
    for cal, tst in (("TinyFace", "SCFace"), ("XQLFW", "SCFace")):
        alpha_rows = []
        for a in ALPHAS_FULL:
            src = "frozen_sensitivity" if any(_close(a, fa) for fa in FROZEN_ALPHAS) else "computed"
            rec = _table_row(sweep, cal, tst, a, src)
            alpha_rows.append(rec)
            table_rows.append(rec)
        table_rows.append(_shift_row(alpha_rows, cal, tst, le20=True))
        table_rows.append(_shift_row(alpha_rows, cal, tst, le20=False))

    write_csv(out_dir() / "table3_robustness.csv", table_rows)
    f4 = fig4_rows(sweep)
    path = out_dir() / "fig4_sensitivity.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=FIG4_FIELDS)
        wtr.writeheader()
        for r in f4:
            wtr.writerow({k: r.get(k, "") for k in FIG4_FIELDS})
    print(f"Wrote {path}", flush=True)
    emit_figure(plot_sensitivity, read_csv(path), "fig_sensitivity")


if __name__ == "__main__":
    main()
