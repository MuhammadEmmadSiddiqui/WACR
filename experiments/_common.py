#!/usr/bin/env python3
"""Shared helpers for paper experiments."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from wacr.jobs import load_wacr
from wacr.protocol import JOBS, W_MAX, load_job
from wacr.score_cache import load_jobs_from_scores, using_scores
from wacr.weights import finalize_wcp_weights, load_importance_weights
from wacr.protocol import _hist_ratio_w, _eval


def load_cfg() -> dict:
    with (REPO / "configs" / "default.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def out_dir() -> Path:
    p = REPO / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fig_dir() -> Path:
    p = REPO / "results" / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}", flush=True)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_jobs():
    wacr = load_wacr()
    if using_scores():
        from wacr.score_cache import enable

        enable()
        return wacr, load_jobs_from_scores(wacr)
    jobs = [load_job(wacr, *spec) for spec in JOBS]
    return wacr, jobs


def hist_w(job: dict, n_bins: int = 20, w_max: float = W_MAX) -> np.ndarray:
    raw = _hist_ratio_w(job["t_cal"], job["t_te"], n_bins=n_bins)
    return finalize_wcp_weights(raw, int(job["n_calib"]), w_max=w_max, epsilon=0.1)


def empty_pct(sizes) -> float:
    return 100.0 * float(np.mean(np.asarray(sizes, dtype=np.float64) <= 0.0))


def eval_conformal(wacr, job, scores, w, alpha: float, *, gamma: float = 1.0, bound: str = "conformal"):
    if bound == "conformal" and abs(float(gamma) - 1.0) > 1e-12:
        tau = wacr.robust_weighted_conformal_threshold(
            scores, w, alpha=alpha, gamma=float(gamma), w_test=W_MAX
        )
        res = wacr.evaluate_adaptive(
            job["test_sims"], tau, job["phi_test"], job["true_test"], min_set_size=0
        )
        valid = job["true_test"] >= 0
        sizes = res.set_sizes[valid].astype(np.float64)
        return float(tau), float(res.r_hat), float(res.avg_set_size), empty_pct(sizes), sizes
    tau, r, c, miss, sizes = _eval(
        wacr, job, scores, w, alpha, True, bound=bound
    )
    return float(tau), float(r), float(c), empty_pct(sizes), sizes


def copy_to_outputs(src: Path, name: str) -> None:
    dest = out_dir() / name
    shutil.copy2(src, dest)
    print(f"Copied {name} -> outputs/", flush=True)


def emit_figure(draw, rows: list[dict], stem: str) -> None:
    png = fig_dir() / f"{stem}.png"
    pdf = fig_dir() / f"{stem}.pdf"
    draw(rows, png, pdf)
    copy_to_outputs(png, png.name)
    copy_to_outputs(pdf, pdf.name)
    from wacr.figures import write_paper_compare

    write_paper_compare(fig_dir(), out_dir() / "compare" / "paper_figures_side_by_side.png")
