#!/usr/bin/env python3
"""Fig 7 / Table 6 weight quality (TV, low-t gap, ESS, range).

Table 6 from the same 1-D weights as make_figure4_weights.py.
Fig 7 CSV is that script's figure4_values collect().
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments._common import emit_figure, hist_w, load_jobs, out_dir, read_csv, write_csv
from wacr.figures import plot_weight_quality
from wacr.protocol import _finalize, _logistic_w, _tv_hist, _tv_low_t
from wacr.weights import effective_sample_size

N_BINS = 20
ORDER = [
    "Unweighted",
    "KMM",
    "KLIEP",
    "Domain classifier",
    "Logistic-linear (ablation)",
    "Logistic-quadratic",
    "WACR (histogram)",
]
EMBED = ("Unweighted", "KMM", "KLIEP", "Domain classifier")
ON_T = ("Logistic-linear (ablation)", "Logistic-quadratic", "WACR (histogram)")
FAMILY = {n: "embedding" for n in EMBED}
FAMILY.update({n: "t" for n in ON_T})
CSV_FIELDS = [
    "kind",
    "pair",
    "cal",
    "weight",
    "family",
    "bin",
    "t_lo",
    "t_hi",
    "density",
    "tv",
    "low_t_gap",
    "ess",
    "ess_pct",
    "w_min",
    "w_max",
    "t_cut",
    "n",
    "note",
]


def collect_weights(job: dict) -> dict[str, np.ndarray]:
    n = job["n_calib"]
    t_cal, t_te = job["t_cal"], job["t_te"]
    out: dict[str, np.ndarray] = {
        "Unweighted": np.ones(n, dtype=np.float64),
        "Logistic-linear (ablation)": _finalize(_logistic_w(t_cal, t_te, 1), n),
        "Logistic-quadratic": _finalize(_logistic_w(t_cal, t_te, 2), n),
        "WACR (histogram)": hist_w(job),
    }
    if job.get("w_kmm") is not None:
        out["KMM"] = np.asarray(job["w_kmm"], dtype=np.float64)
    if job.get("w_kliep") is not None:
        out["KLIEP"] = np.asarray(job["w_kliep"], dtype=np.float64)
    if job.get("w_dc") is not None and job["cal"] == "TinyFace":
        out["Domain classifier"] = np.asarray(job["w_dc"], dtype=np.float64)
    return out


def wacr_edges(t_cal: np.ndarray, t_te: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    lo = float(min(float(t_cal.min()), float(t_te.min())))
    hi = float(max(float(t_cal.max()), float(t_te.max())))
    return np.linspace(lo, hi, n_bins + 1)


def bin_density(t: np.ndarray, edges: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is None:
        hc, _ = np.histogram(t, bins=edges)
    else:
        hc, _ = np.histogram(t, bins=edges, weights=np.asarray(w, dtype=np.float64))
    width = np.diff(edges)
    s = float(hc.sum())
    dens = np.zeros(hc.shape, dtype=np.float64)
    if s > 0:
        dens = hc / (s * width)
    return dens


def empty_row(**kw) -> dict:
    rec = {k: "" for k in CSV_FIELDS}
    rec.update(kw)
    return rec


def fig7_rows(jobs) -> list[dict]:
    rows: list[dict] = []
    for job in jobs:
        cal = job["cal"]
        pair = f"{cal} -> SCFace"
        t_cal = np.asarray(job["t_cal"], dtype=np.float64).reshape(-1)
        t_te = np.asarray(job["t_te"], dtype=np.float64).reshape(-1)
        t_cut = float(np.median(t_te))
        edges = wacr_edges(t_cal, t_te)
        wmap = collect_weights(job)
        dens_te = bin_density(t_te, edges)
        for b in range(N_BINS):
            rows.append(
                empty_row(
                    kind="target",
                    pair=pair,
                    cal=cal,
                    weight="SCFace",
                    family="target",
                    bin=b,
                    t_lo=f"{edges[b]:.10g}",
                    t_hi=f"{edges[b + 1]:.10g}",
                    density=f"{dens_te[b]:.10g}",
                    t_cut=f"{t_cut:.10g}",
                    n=int(t_te.size),
                )
            )
        for name in ORDER:
            if name not in wmap:
                rows.append(
                    empty_row(
                        kind="summary",
                        pair=pair,
                        cal=cal,
                        weight=name,
                        family=FAMILY[name],
                        n=int(t_cal.size),
                        t_cut=f"{t_cut:.10g}",
                        note="unavailable",
                    )
                )
                continue
            w = np.asarray(wmap[name], dtype=np.float64).reshape(-1)
            ess = float(effective_sample_size(w))
            tv = float(_tv_hist(t_cal, w, t_te, n_bins=N_BINS))
            gap = float(_tv_low_t(t_cal, w, t_te, t_cut))
            rows.append(
                empty_row(
                    kind="summary",
                    pair=pair,
                    cal=cal,
                    weight=name,
                    family=FAMILY[name],
                    tv=f"{tv:.10g}",
                    low_t_gap=f"{gap:.10g}",
                    ess=f"{ess:.10g}",
                    ess_pct=f"{100.0 * ess / max(t_cal.size, 1):.10g}",
                    w_min=f"{float(w.min()):.10g}",
                    w_max=f"{float(w.max()):.10g}",
                    t_cut=f"{t_cut:.10g}",
                    n=int(t_cal.size),
                )
            )
            dens = bin_density(t_cal, edges, w)
            for b in range(N_BINS):
                rows.append(
                    empty_row(
                        kind="density",
                        pair=pair,
                        cal=cal,
                        weight=name,
                        family=FAMILY[name],
                        bin=b,
                        t_lo=f"{edges[b]:.10g}",
                        t_hi=f"{edges[b + 1]:.10g}",
                        density=f"{dens[b]:.10g}",
                    )
                )
    return rows


def main() -> None:
    _, jobs = load_jobs()
    rows = []
    for job in jobs:
        t_cal, t_te = job["t_cal"], job["t_te"]
        t_cut = float(np.median(t_te))
        n = job["n_calib"]
        weights = collect_weights(job)
        for name in ORDER:
            if name not in weights:
                rows.append(
                    {
                        "pair": job["cal"],
                        "weighting": name,
                        "tv": "",
                        "low_t_gap": "",
                        "ess_pct": "",
                        "w_min": "",
                        "w_max": "",
                    }
                )
                continue
            w = np.asarray(weights[name], dtype=np.float64)
            ess = effective_sample_size(w)
            rows.append(
                {
                    "pair": job["cal"],
                    "weighting": name,
                    "tv": float(_tv_hist(t_cal, w, t_te)),
                    "low_t_gap": float(_tv_low_t(t_cal, w, t_te, t_cut)),
                    "ess_pct": 100.0 * ess / n,
                    "w_min": float(w.min()),
                    "w_max": float(w.max()),
                }
            )
            print(
                f"{job['cal']} {name} TV={rows[-1]['tv']:.4f} ESS%={rows[-1]['ess_pct']:.1f}",
                flush=True,
            )
    write_csv(out_dir() / "table6_weight_quality.csv", rows)

    f7 = fig7_rows(jobs)
    path = out_dir() / "fig7_weight_quality.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wtr.writeheader()
        for r in f7:
            wtr.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"Wrote {path}", flush=True)
    emit_figure(plot_weight_quality, read_csv(path), "fig_weight_quality")


if __name__ == "__main__":
    main()
