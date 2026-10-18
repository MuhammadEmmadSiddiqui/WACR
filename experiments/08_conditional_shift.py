#!/usr/bin/env python3
"""Table 7 / Fig 8 conditional invariance (S given t).

Copied from Outputs/make_figure3_a1.py collect().
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments._common import emit_figure, load_jobs, out_dir, read_csv
from wacr.figures import plot_conditional
from wacr.protocol import BETA, W_MAX, _finalize, _group_index, _hist_ratio_w, _seed_for

ALPHA = 0.10
GAMMA = 1.0
N_BINS = 20
MIN_BIN = 20
N_BOOT = 1000
SEED = 0
CSV_FIELDS = [
    "kind",
    "panel",
    "pair",
    "domain",
    "bin",
    "t_lo",
    "t_hi",
    "t_mean",
    "n",
    "p",
    "lo",
    "hi",
    "tau",
    "r_hat",
    "r_w",
    "delta",
    "note",
]


def wacr_edges(t_a: np.ndarray, t_b: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    lo = float(min(float(t_a.min()), float(t_b.min())))
    hi = float(max(float(t_a.max()), float(t_b.max())))
    return np.linspace(lo, hi, n_bins + 1)


def strat_bin_curve(t, y, ids, edges, rng, n_boot=N_BOOT, min_bin=MIN_BIN):
    n_bins = len(edges) - 1
    idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    inside = (t >= edges[0]) & (t <= edges[-1])
    t_mean = np.full(n_bins, np.nan)
    p = np.full(n_bins, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    n = np.zeros(n_bins, dtype=np.int64)
    groups = _group_index(np.asarray(ids))
    parts = [g[rng.integers(0, g.size, size=(n_boot, g.size))] for g in groups]
    boot_idx = np.concatenate(parts, axis=1)
    y = np.asarray(y, dtype=np.float64)
    for b in range(n_bins):
        m = inside & (idx == b)
        n[b] = int(m.sum())
        if n[b] < min_bin:
            continue
        t_mean[b] = float(t[m].mean())
        p[b] = float(y[m].mean())
        mb = m[boot_idx]
        yb = y[boot_idx]
        counts = mb.sum(axis=1)
        num = (yb * mb).sum(axis=1)
        means = np.full(n_boot, np.nan)
        okc = counts >= 8
        means[okc] = num[okc] / counts[okc]
        ok = np.isfinite(means)
        if int(ok.sum()) < 20:
            continue
        lo[b] = float(np.quantile(means[ok], 0.025))
        hi[b] = float(np.quantile(means[ok], 0.975))
    return t_mean, p, lo, hi, n


def integrated_gap(t_src, y_src, t_te, y_te, edges) -> float:
    n_bins = len(edges) - 1
    is_ = np.clip(np.digitize(t_src, edges) - 1, 0, n_bins - 1)
    it = np.clip(np.digitize(t_te, edges) - 1, 0, n_bins - 1)
    in_s = (t_src >= edges[0]) & (t_src <= edges[-1])
    in_t = (t_te >= edges[0]) & (t_te <= edges[-1])
    nt = np.array([int(((it == b) & in_t).sum()) for b in range(n_bins)], dtype=np.float64)
    p_te = nt / max(float(nt.sum()), 1.0)
    delta = 0.0
    for b in range(n_bins):
        ms = in_s & (is_ == b)
        mt = in_t & (it == b)
        if int(ms.sum()) == 0 or int(mt.sum()) == 0:
            continue
        ps = float(y_src[ms].mean())
        pt = float(y_te[mt].mean())
        delta += float(p_te[b]) * (pt - ps)
    return float(delta)


def empty_row(**kw) -> dict:
    rec = {k: "" for k in CSV_FIELDS}
    rec.update(kw)
    return rec


def load_scface_eval(job: dict):
    sims = job["test_sims"]
    true = np.asarray(job["true_test"])
    valid = true >= 0
    rows = np.where(valid)[0]
    t = sims[rows].max(axis=1)
    cos = sims[rows, true[rows]]
    phi = np.clip(1.0 - t, 0.0, 1.0) + BETA
    s = (1.0 - cos) / phi
    hit = (cos >= t).astype(np.float64)
    ids = np.asarray(job["test_ids"])
    return t, s, hit, ids


def collect(wacr, jobs) -> list[dict]:
    jobs_by = {j["cal"]: j for j in jobs}
    series = {}
    for job in jobs:
        t = np.asarray(job["t_cal"], dtype=np.float64).reshape(-1)
        s = np.asarray(job["s_g"], dtype=np.float64).reshape(-1)
        cos = np.asarray(job["cos_true"], dtype=np.float64).reshape(-1)
        series[job["cal"]] = {
            "t": t,
            "s": s,
            "hit": (cos >= t).astype(np.float64),
            "ids": np.asarray(job["calib_ids"]),
        }

    t_te, s_te, hit_te, ids_te = load_scface_eval(jobs[0])
    series["SCFace"] = {"t": t_te, "s": s_te, "hit": hit_te, "ids": ids_te}

    facts = {}
    rows: list[dict] = []
    for cal in ("TinyFace", "XQLFW"):
        job = jobs_by[cal]
        t_cal = series[cal]["t"]
        s_cal = series[cal]["s"]
        w = _finalize(_hist_ratio_w(t_cal, job["t_te"], n_bins=N_BINS), int(t_cal.size))
        tau = wacr.robust_weighted_conformal_threshold(
            s_cal, w, alpha=ALPHA, gamma=GAMMA, w_test=W_MAX
        )
        res = wacr.evaluate_adaptive(
            job["test_sims"], tau, job["phi_test"], job["true_test"], min_set_size=0
        )
        valid = job["true_test"] >= 0
        r_hat = float((~res.contains_true[valid]).mean())
        exc_cal = (s_cal > tau).astype(np.float64)
        r_w = float(np.average(exc_cal, weights=w))
        exc_te = (s_te > tau).astype(np.float64)
        edges = wacr_edges(t_cal, job["t_te"])
        delta = integrated_gap(t_cal, exc_cal, t_te, exc_te, edges)
        facts[cal] = {
            "tau": float(tau),
            "r_hat": r_hat,
            "r_w": r_w,
            "delta": delta,
            "edges": edges,
            "exc_cal": exc_cal,
            "exc_te": exc_te,
            "w": w,
        }
        print(
            f"{cal} -> SCFace  tau={tau:.6f}  R-hat={r_hat:.4f}  R_w={r_w:.4f}  "
            f"Delta={delta:+.4f}",
            flush=True,
        )
        rows.append(
            empty_row(
                kind="summary",
                panel="b" if cal == "TinyFace" else "c",
                pair=f"{cal} -> SCFace",
                tau=f"{tau:.10g}",
                r_hat=f"{r_hat:.10g}",
                r_w=f"{r_w:.10g}",
                delta=f"{delta:.10g}",
                note="Method B hist conformal Gamma=1 a=0.10",
            )
        )

    edges_a = wacr_edges(
        np.concatenate([series[n]["t"] for n in ("TinyFace", "XQLFW", "SCFace")]),
        series["SCFace"]["t"],
    )

    def add_curve(panel, pair, domain, t, y, ids, edges, kind):
        rng = np.random.default_rng(_seed_for(SEED, "fig3", panel, domain, kind))
        tm, p, lo, hi, n = strat_bin_curve(t, y, ids, edges, rng)
        for b in range(len(n)):
            rows.append(
                empty_row(
                    kind=kind,
                    panel=panel,
                    pair=pair,
                    domain=domain,
                    bin=b,
                    t_lo=f"{edges[b]:.10g}",
                    t_hi=f"{edges[b + 1]:.10g}",
                    t_mean="" if not np.isfinite(tm[b]) else f"{tm[b]:.10g}",
                    n=int(n[b]),
                    p="" if not np.isfinite(p[b]) else f"{p[b]:.10g}",
                    lo="" if not np.isfinite(lo[b]) else f"{lo[b]:.10g}",
                    hi="" if not np.isfinite(hi[b]) else f"{hi[b]:.10g}",
                )
            )

    for name in ("TinyFace", "XQLFW", "SCFace"):
        ser = series[name]
        add_curve("a", "all", name, ser["t"], ser["hit"], ser["ids"], edges_a, "hit")

    add_curve(
        "b",
        "TinyFace -> SCFace",
        "TinyFace",
        series["TinyFace"]["t"],
        facts["TinyFace"]["exc_cal"],
        series["TinyFace"]["ids"],
        facts["TinyFace"]["edges"],
        "exc",
    )
    add_curve(
        "b",
        "TinyFace -> SCFace",
        "SCFace",
        series["SCFace"]["t"],
        facts["TinyFace"]["exc_te"],
        series["SCFace"]["ids"],
        facts["TinyFace"]["edges"],
        "exc",
    )
    add_curve(
        "c",
        "XQLFW -> SCFace",
        "XQLFW",
        series["XQLFW"]["t"],
        facts["XQLFW"]["exc_cal"],
        series["XQLFW"]["ids"],
        facts["XQLFW"]["edges"],
        "exc",
    )
    add_curve(
        "c",
        "XQLFW -> SCFace",
        "SCFace",
        series["SCFace"]["t"],
        facts["XQLFW"]["exc_te"],
        series["SCFace"]["ids"],
        facts["XQLFW"]["edges"],
        "exc",
    )

    for cal, panel in (("TinyFace", "b"), ("XQLFW", "c")):
        edges = facts[cal]["edges"]
        t = series["SCFace"]["t"]
        n_bins = len(edges) - 1
        idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
        inside = (t >= edges[0]) & (t <= edges[-1])
        width = float(edges[1] - edges[0])
        for b in range(n_bins):
            m = inside & (idx == b)
            dens = float(m.sum()) / (max(t.size, 1) * width)
            rows.append(
                empty_row(
                    kind="density",
                    panel=panel,
                    pair=f"{cal} -> SCFace",
                    domain="SCFace",
                    bin=b,
                    t_lo=f"{edges[b]:.10g}",
                    t_hi=f"{edges[b + 1]:.10g}",
                    t_mean=f"{0.5 * (edges[b] + edges[b + 1]):.10g}",
                    n=int(m.sum()),
                    p=f"{dens:.10g}",
                )
            )

    for name in ("TinyFace", "XQLFW", "SCFace"):
        t = series[name]["t"]
        hit = series[name]["hit"]
        m = (t >= 0.182) & (t < 0.248)
        rows.append(
            empty_row(
                kind="sanity_hit",
                panel="a",
                pair="all",
                domain=name,
                t_lo="0.182",
                t_hi="0.248",
                t_mean="" if int(m.sum()) == 0 else f"{float(t[m].mean()):.10g}",
                n=int(m.sum()),
                p="" if int(m.sum()) == 0 else f"{float(hit[m].mean()):.10g}",
                note="window around t=0.21 (10-bin diagnostic)",
            )
        )
    return rows


def main() -> None:
    wacr, jobs = load_jobs()
    rows = collect(wacr, jobs)
    path = out_dir() / "table7_conditional.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"Wrote {path}", flush=True)
    emit_figure(plot_conditional, read_csv(path), "fig_conditional")


if __name__ == "__main__":
    main()
