#!/usr/bin/env python3
"""Validity protocol for gallery-matched WACR (conformal / Bernstein / APUB).

Estimand: closed-set miss risk on a fixed SCFace gallery (IDs 066-130).
Identities are strata. Do not add IDs 001-065.

1. Retune KMM once with the median RBF heuristic; report whether it tracks t.
2. Full-set conformal: Unweighted, hist, quadratic, top-1 LR, and KMM median
   (if computed). Methods A and B, tinyface_scface and xqlfw_scface, alpha in {0.05,0.10,0.20}.
   Identity-stratified bootstrap CI on R-hat - alpha (probes within ID);
   ID-cluster bootstrap as sensitivity. Paired Delta vs hist on the same resamples.
3. ~50 SCFace identity-stratified fit/eval splits. Weights on the unlabeled
   fit-half. Mean +/- sd of R-hat and |C| (stability only, not a CI for E[R]).
4. Bernstein / APUB: cluster-resample calibration identities, redraw matched
   impostors, refit 1-D weights, report the fraction of draws with R-hat > alpha.

KMM is not looped in (3)-(4) unless it tracks t (then full-set only still applies;
splits/PAC skip KMM because the QP is too expensive).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from wacr.jobs import align_gmatch, load_wacr  # noqa: E402
from wacr.paths import pair_cache_dir  # noqa: E402
from wacr.similarity import l2_normalize  # noqa: E402
from wacr.weights import (  # noqa: E402
    effective_sample_size,
    finalize_wcp_weights,
    load_importance_weights,
)


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gmatch = _load_mod(
    "gmatch_cal", REPO_ROOT / "wacr" / "galleries.py"
)
_kmm = _load_mod("kmm_mod", REPO_ROOT / "baselines" / "wcp.py")
matched_gallery_top1 = _gmatch.matched_gallery_top1
get_kmm_weights = _kmm.get_kmm_weights
median_rbf_gamma = _kmm.median_rbf_gamma

ALPHAS = (0.05, 0.10, 0.20)
JOBS = (
    ("tinyface_scface", "TinyFace", "SCFace"),
    ("xqlfw_scface", "XQLFW", "SCFace"),
)
W_MAX = 30.0
BETA = 0.5
DELTA = 0.1
FIG = REPO_ROOT / "Outputs" / "figures"
CORE_WEIGHTS = ("Unweighted", "hist ratio", "quadratic LR", "top-1 LR")


def _seed_for(*parts) -> int:
    h = 2166136261
    for b in "|".join(map(str, parts)).encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _group_index(ids: np.ndarray) -> list[np.ndarray]:
    uniq, inv = np.unique(ids, return_inverse=True)
    return [np.flatnonzero(inv == i) for i in range(len(uniq))]


def _logistic_w(t_cal: np.ndarray, t_test: np.ndarray, degree: int) -> np.ndarray:
    x_cal = t_cal.reshape(-1, 1)
    x_test = t_test.reshape(-1, 1)
    x = np.vstack([x_cal, x_test])
    y = np.r_[np.zeros(len(x_cal)), np.ones(len(x_test))]
    clf = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000),
    ).fit(x, y)
    p = clf.predict_proba(x_cal)[:, 1]
    return (p / (1.0 - p + 1e-12)) * (len(x_cal) / len(x_test))


def _hist_ratio_w(t_cal: np.ndarray, t_test: np.ndarray, n_bins: int = 20) -> np.ndarray:
    lo = float(min(t_cal.min(), t_test.min()))
    hi = float(max(t_cal.max(), t_test.max()))
    edges = np.linspace(lo, hi, n_bins + 1)
    hc, _ = np.histogram(t_cal, bins=edges)
    ht, _ = np.histogram(t_test, bins=edges)
    pc = (hc + 0.5) / (hc.sum() + 0.5 * n_bins)
    pt = (ht + 0.5) / (ht.sum() + 0.5 * n_bins)
    idx = np.clip(np.digitize(t_cal, edges) - 1, 0, n_bins - 1)
    return (pt / pc)[idx]


def _finalize(w_raw: np.ndarray, n: int) -> np.ndarray:
    return finalize_wcp_weights(w_raw, n, w_max=W_MAX, epsilon=0.1)


def _tv_low_t(t_cal: np.ndarray, w: np.ndarray, t_te: np.ndarray, t_cut: float) -> float:
    edges = np.array(
        [
            float(min(t_cal.min(), t_te.min())),
            t_cut,
            float(max(t_cal.max(), t_te.max())),
        ]
    )
    hc, _ = np.histogram(t_cal, bins=edges, weights=w / w.sum())
    ht, _ = np.histogram(t_te, bins=edges, density=False)
    ht = ht / max(ht.sum(), 1.0)
    return float(hc[0] - ht[0])


def _tv_hist(t_cal: np.ndarray, w: np.ndarray, t_te: np.ndarray, n_bins: int = 20) -> float:
    lo = float(min(t_cal.min(), t_te.min()))
    hi = float(max(t_cal.max(), t_te.max()))
    edges = np.linspace(lo, hi, n_bins + 1)
    hc, _ = np.histogram(t_cal, bins=edges, weights=w)
    ht, _ = np.histogram(t_te, bins=edges)
    pc = hc / max(float(hc.sum()), 1e-12)
    pt = ht / max(float(ht.sum()), 1e-12)
    return float(0.5 * np.abs(pc - pt).sum())


def _build_1d_weights(
    t_cal: np.ndarray, t_te: np.ndarray, names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    n = int(t_cal.shape[0])
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name == "Unweighted":
            raw = np.ones(n, dtype=np.float64)
        elif name == "hist ratio":
            raw = _hist_ratio_w(t_cal, t_te)
        elif name == "quadratic LR":
            raw = _logistic_w(t_cal, t_te, 2)
        elif name == "top-1 LR":
            raw = _logistic_w(t_cal, t_te, 1)
        else:
            raise ValueError(name)
        out[name] = _finalize(raw, n)
    return out


def load_job(wacr, comb: str, cal: str, tst: str) -> dict:
    cache = pair_cache_dir(comb)
    calib = wacr.load_bundle(cache / f"{cal}_Queries_mcd.npz", name=f"{cal}_Queries")
    test = wacr.load_bundle(cache / f"{tst}_Queries_mcd.npz", name=f"{tst}_Queries")
    cg = wacr.load_bundle(cache / f"{cal}_Gallery_mcd.npz", name=f"{cal}_Gallery")
    tg = wacr.load_bundle(cache / f"{tst}_Gallery_mcd.npz", name=f"{tst}_Gallery")
    true_calib = wacr.true_match_indices(calib, cg)
    true_test = wacr.true_match_indices(test, tg)
    valid = true_calib >= 0
    if not np.all(valid):
        calib = replace(
            calib,
            embeddings=calib.embeddings[valid],
            paths=[p for p, v in zip(calib.paths, valid) if v],
            fu=(calib.fu[valid] if calib.fu is not None else None),
        )
        true_calib = true_calib[valid]
    calib_sims = wacr.cosine_similarity_matrix(calib.embeddings, cg.embeddings)
    test_sims = wacr.cosine_similarity_matrix(test.embeddings, tg.embeddings)
    s_base = (1.0 - calib_sims[np.arange(calib_sims.shape[0]), true_calib]).astype(
        np.float64
    )
    phi_cal, phi_test = wacr._fit_phi(
        uncertainty_method="top1",
        beta=BETA,
        calib=calib,
        test=test,
        calib_sims=calib_sims,
        test_sims=test_sims,
    )
    gpath = cache / f"{cal}_Queries_gmatch.npz"
    gz = np.load(gpath, allow_pickle=True)
    gpaths = [str(p) for p in gz["paths"]]
    cal_paths = list(calib.paths)

    def _align_g(arr) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        if gpaths == cal_paths:
            return a
        idx = {p: i for i, p in enumerate(gpaths)}
        return np.array([a[idx[p]] for p in cal_paths], dtype=np.float64)

    t_cal = _align_g(gz["top1_matched"])
    phi_matched = _align_g(gz["phi_matched"])
    s_g = align_gmatch(gpath, cal_paths)
    cos_true = _align_g(gz["cos_true"])
    top1_full = _align_g(gz["top1_full"])
    test_valid = true_test >= 0
    rec = {
        "comb": comb,
        "cal": cal,
        "tst": tst,
        "cache": cache,
        "calib_paths": list(calib.paths),
        "calib_ids": np.asarray(calib.identities),
        "calib_emb": np.asarray(calib.embeddings, dtype=np.float64),
        "calib_sims": calib_sims,
        "cg_ids": np.asarray(cg.identities),
        "s_base": s_base,
        "s_g": s_g,
        "t_cal": t_cal,
        "phi_matched": phi_matched,
        "phi_cal": phi_cal,
        "test_ids": np.asarray(test.identities)[test_valid],
        "test_qids": np.asarray(test.identities),
        "tg_ids": np.asarray(tg.identities),
        "test_paths": list(test.paths),
        "test_emb": np.asarray(test.embeddings, dtype=np.float64),
        "test_sims": test_sims,
        "true_test": true_test,
        "phi_test": phi_test,
        "t_te": test_sims.max(axis=1),
        "cos_true": cos_true,
        "top1_full": top1_full,
        "n_eval": int(test_valid.sum()),
        "n_calib": int(s_base.shape[0]),
        "n_ids_test": len(set(tg.identities)),
        "test_valid": test_valid,
        "from_scores": False,
    }
    for tag, key in (("kmm_median", "w_kmm"), ("kliep", "w_kliep"), ("wx", "w_dc")):
        p = cache / f"{cal}_Queries_{tag}.npz"
        rec[key] = load_importance_weights(p, rec["calib_paths"]) if p.is_file() else None
    return rec


def _eval(
    wacr,
    job: dict,
    scores: np.ndarray,
    w: np.ndarray,
    alpha: float,
    adaptive: bool,
    *,
    bound: str = "conformal",
    test_idx: np.ndarray | None = None,
    apub_lookup=None,
):
    if bound == "conformal":
        tau = wacr.weighted_conformal_threshold(scores, w, alpha=alpha, w_test=W_MAX)
    elif bound in {"bernstein", "hoeffding"}:
        tau, _ = wacr.concentration_threshold(
            scores, w, alpha=alpha, delta=DELTA, kind=bound
        )
    elif bound == "apub":
        if apub_lookup is None:
            tau, _ = wacr.apub_threshold(scores, w, alpha=alpha, delta=DELTA)
        else:
            unique_tau, apub = apub_lookup
            hit = np.flatnonzero(apub <= alpha)
            tau = float(unique_tau[int(hit[0])] if hit.size else float("inf"))
    else:
        raise ValueError(bound)

    sims = job["test_sims"]
    phi = job["phi_test"]
    true = job["true_test"]
    if test_idx is not None:
        sims = sims[test_idx]
        phi = phi[test_idx]
        true = true[test_idx]
    if adaptive:
        res = wacr.evaluate_adaptive(sims, tau, phi, true, min_set_size=0)
    else:
        res = wacr.evaluate_baseline(sims, tau, true, min_set_size=0)
    valid = true >= 0
    miss = (~res.contains_true[valid]).astype(np.float64)
    sizes = res.set_sizes[valid].astype(np.float64)
    return float(tau), float(res.r_hat), float(res.avg_set_size), miss, sizes


def _seed_for(*parts) -> int:
    h = 2166136261
    for b in "|".join(map(str, parts)).encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _group_index(ids: np.ndarray) -> list[np.ndarray]:
    uniq, inv = np.unique(ids, return_inverse=True)
    return [np.flatnonzero(inv == i) for i in range(len(uniq))]


def _strat_boot_means(miss: np.ndarray, groups: list[np.ndarray], rng, n_boot: int) -> np.ndarray:
    parts = []
    for g in groups:
        draw = rng.integers(0, g.size, size=(n_boot, g.size))
        parts.append(miss[g][draw])
    return np.concatenate(parts, axis=1).mean(axis=1)


def _cluster_boot_means(miss: np.ndarray, groups: list[np.ndarray], rng, n_boot: int) -> np.ndarray:
    k = len(groups)
    id_mean = np.array([float(miss[g].mean()) for g in groups], dtype=np.float64)
    id_n = np.array([float(g.size) for g in groups], dtype=np.float64)
    pick = rng.integers(0, k, size=(n_boot, k))
    w = id_n[pick]
    return (w * id_mean[pick]).sum(axis=1) / w.sum(axis=1)


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"Wrote {path}", flush=True)


def _kmm_once(job: dict) -> tuple[np.ndarray | None, dict]:
    x_cal = l2_normalize(job["calib_emb"])
    x_te = l2_normalize(job["test_emb"])
    stacked = np.vstack([x_cal, x_te])
    gamma = median_rbf_gamma(stacked)
    sklearn_g = 1.0 / float(x_cal.shape[1])
    diag = {
        "combination": job["comb"],
        "cal": job["cal"],
        "gamma_median": gamma,
        "gamma_sklearn": sklearn_g,
        "status": "ok",
        "tracks_t": 0,
        "ESS": float("nan"),
        "ESS_over_n": float("nan"),
        "tv": float("nan"),
        "low_t_excess": float("nan"),
        "tv_hist": float("nan"),
        "tv_unweighted": float("nan"),
        "low_t_hist": float("nan"),
        "w_min": float("nan"),
        "w_max": float("nan"),
    }
    print(
        f"  KMM median gamma={gamma:.6g} (sklearn default {sklearn_g:.6g})",
        flush=True,
    )
    try:
        raw = get_kmm_weights(x_cal, x_te, B=W_MAX, epsilon=0.1, gamma=gamma)
    except Exception as exc:
        diag["status"] = f"fail: {type(exc).__name__}: {exc}"
        print(f"  KMM median failed: {diag['status']}", flush=True)
        return None, diag
    w = _finalize(np.asarray(raw, dtype=np.float64).reshape(-1), job["n_calib"])
    t_cut = float(np.median(job["t_te"]))
    w_hist = _finalize(_hist_ratio_w(job["t_cal"], job["t_te"]), job["n_calib"])
    w_unif = np.ones(job["n_calib"], dtype=np.float64)
    diag["ESS"] = float(effective_sample_size(w))
    diag["ESS_over_n"] = diag["ESS"] / job["n_calib"]
    diag["tv"] = _tv_hist(job["t_cal"], w, job["t_te"])
    diag["tv_hist"] = _tv_hist(job["t_cal"], w_hist, job["t_te"])
    diag["tv_unweighted"] = _tv_hist(job["t_cal"], w_unif, job["t_te"])
    diag["low_t_excess"] = _tv_low_t(job["t_cal"], w, job["t_te"], t_cut)
    diag["low_t_hist"] = _tv_low_t(job["t_cal"], w_hist, job["t_te"], t_cut)
    diag["w_min"] = float(w.min())
    diag["w_max"] = float(w.max())
    closer = diag["tv"] <= 0.5 * (diag["tv_hist"] + diag["tv_unweighted"])
    low_ok = abs(diag["low_t_excess"]) <= abs(diag["low_t_hist"]) + 0.03
    diag["tracks_t"] = int(closer or low_ok)
    print(
        "  KMM median ESS=%.0f (%.1f%%)  TV=%.3f (hist %.3f, unif %.3f)  "
        "low-t=%+.3f (hist %+.3f)  tracks_t=%s"
        % (
            diag["ESS"],
            100.0 * diag["ESS_over_n"],
            diag["tv"],
            diag["tv_hist"],
            diag["tv_unweighted"],
            diag["low_t_excess"],
            diag["low_t_hist"],
            "yes" if diag["tracks_t"] else "no",
        ),
        flush=True,
    )
    out_path = job["cache"] / f"{job['cal']}_Queries_kmm_median.npz"
    np.savez_compressed(
        out_path,
        w=w.astype(np.float32),
        paths=np.array(job["calib_paths"]),
        method=np.array("standard"),
        B=np.float32(W_MAX),
        epsilon=np.float32(0.1),
        gamma=np.float32(gamma),
        gamma_spec=np.array("median"),
    )
    print(f"  saved {out_path}", flush=True)
    return w, diag


def _identity_split(ids: np.ndarray, rng, frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    fit = np.zeros(ids.shape[0], dtype=bool)
    for g in _group_index(ids):
        n = int(g.size)
        if n == 1:
            fit[g] = bool(rng.random() < frac)
            continue
        n_fit = int(round(frac * n))
        n_fit = min(max(1, n_fit), n - 1)
        fit[rng.choice(g, size=n_fit, replace=False)] = True
    ev = ~fit
    if not ev.any():
        ev[rng.integers(0, ids.shape[0])] = True
        fit[ev] = False
    if not fit.any():
        fit[rng.integers(0, ids.shape[0])] = True
        ev[fit] = False
    return fit, ev


def _calib_cluster_idx(calib_ids: np.ndarray, rng) -> np.ndarray:
    groups = _group_index(calib_ids)
    pick = rng.choice(len(groups), size=len(groups), replace=True)
    return np.concatenate([groups[i] for i in pick])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-splits", type=int, default=50)
    p.add_argument("--n-pac", type=int, default=30)
    p.add_argument("--skip-kmm", action="store_true")
    p.add_argument("--skip-splits", action="store_true")
    p.add_argument("--skip-pac", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wacr = load_wacr()
    kmm_rows: list[dict] = []
    full_rows: list[dict] = []
    pair_rows: list[dict] = []
    split_rows: list[dict] = []
    pac_rows: list[dict] = []
    split_sum: list[dict] = []
    pac_sum: list[dict] = []

    jobs = [load_job(wacr, *spec) for spec in JOBS]
    weights_full: dict[str, dict[str, np.ndarray]] = {}
    include_kmm: dict[str, bool] = {}

    for job in jobs:
        comb = job["comb"]
        print(f"\n=== {comb}: {job['cal']} -> {job['tst']}  n_cal={job['n_calib']}  "
              f"n_eval={job['n_eval']}  k={job['n_ids_test']} ===", flush=True)
        wmap = _build_1d_weights(job["t_cal"], job["t_te"], CORE_WEIGHTS)
        include_kmm[comb] = False
        if not args.skip_kmm:
            w_kmm, kdiag = _kmm_once(job)
            kmm_rows.append(kdiag)
            if w_kmm is not None:
                wmap["KMM median"] = w_kmm
                include_kmm[comb] = bool(kdiag["tracks_t"])
        weights_full[comb] = wmap

        groups = _group_index(job["test_ids"])
        miss_store: dict[tuple[str, str, float], np.ndarray] = {}
        for wname, w in wmap.items():
            for meth, scores, adaptive in (
                ("A", job["s_base"], False),
                ("B", job["s_g"], True),
            ):
                for a in ALPHAS:
                    tau, r, c, miss, _sizes = _eval(
                        wacr, job, scores, w, a, adaptive, bound="conformal"
                    )
                    miss_store[(wname, meth, a)] = miss
                    rng = np.random.default_rng(_seed_for(args.seed, comb, wname, meth, a, "full"))
                    strat = _strat_boot_means(miss, groups, rng, args.n_boot)
                    clus = _cluster_boot_means(miss, groups, rng, args.n_boot)
                    d_s = strat - a
                    d_c = clus - a
                    row = {
                        "combination": comb,
                        "cal": job["cal"],
                        "weight": wname,
                        "method": meth,
                        "alpha": a,
                        "bound": "conformal",
                        "tau": tau,
                        "r_hat": r,
                        "set_size": c,
                        "r_minus_a": r - a,
                        "controlled": int(r <= a),
                        "n_eval": job["n_eval"],
                        "n_boot": args.n_boot,
                        "strat_lo": _pct(d_s, 0.025),
                        "strat_hi": _pct(d_s, 0.975),
                        "strat_covers_0": int(_pct(d_s, 0.025) <= 0.0 <= _pct(d_s, 0.975)),
                        "cluster_lo": _pct(d_c, 0.025),
                        "cluster_hi": _pct(d_c, 0.975),
                        "cluster_covers_0": int(
                            _pct(d_c, 0.025) <= 0.0 <= _pct(d_c, 0.975)
                        ),
                    }
                    full_rows.append(row)
                    print(
                        "  %s %s a=%.2f  R=%.4f |C|=%.1f  R-a=%.4f  "
                        "strat[%.3f, %.3f]  clust[%.3f, %.3f]  %s"
                        % (
                            wname,
                            meth,
                            a,
                            r,
                            c,
                            r - a,
                            row["strat_lo"],
                            row["strat_hi"],
                            row["cluster_lo"],
                            row["cluster_hi"],
                            "OK" if r <= a else "FAIL",
                        ),
                        flush=True,
                    )

        for wname in wmap:
            if wname == "hist ratio":
                continue
            for meth in ("A", "B"):
                for a in ALPHAS:
                    m = miss_store[(wname, meth, a)]
                    h = miss_store[("hist ratio", meth, a)]
                    rng = np.random.default_rng(_seed_for(args.seed, comb, wname, meth, a, "pair"))
                    d_s = _strat_boot_means(m - h, groups, rng, args.n_boot)
                    d_c = _cluster_boot_means(m - h, groups, rng, args.n_boot)
                    pair_rows.append(
                        {
                            "combination": comb,
                            "cal": job["cal"],
                            "weight": wname,
                            "method": meth,
                            "alpha": a,
                            "delta": float((m - h).mean()),
                            "strat_lo": _pct(d_s, 0.025),
                            "strat_hi": _pct(d_s, 0.975),
                            "cluster_lo": _pct(d_c, 0.025),
                            "cluster_hi": _pct(d_c, 0.975),
                            "frac_strat_pos": float((d_s > 0).mean()),
                        }
                    )

        if args.skip_splits:
            continue
        print(f"  SCFace splits n={args.n_splits} ...", flush=True)
        split_names = CORE_WEIGHTS
        acc: dict[tuple[str, str, float], list[tuple[float, float]]] = {}
        for s in range(args.n_splits):
            rng = np.random.default_rng(args.seed + 1000 + s)
            fit_m, ev_m = _identity_split(job["test_ids"], rng)
            # test_ids is already restricted to valid queries; map back to full rows
            valid_idx = np.flatnonzero(job["test_valid"])
            fit_full = valid_idx[fit_m]
            ev_full = valid_idx[ev_m]
            t_fit = job["t_te"][fit_full]
            wmap_s = _build_1d_weights(job["t_cal"], t_fit, split_names)
            for wname, w in wmap_s.items():
                for meth, scores, adaptive in (
                    ("A", job["s_base"], False),
                    ("B", job["s_g"], True),
                ):
                    for a in ALPHAS:
                        _tau, r, c, _miss, _sz = _eval(
                            wacr,
                            job,
                            scores,
                            w,
                            a,
                            adaptive,
                            bound="conformal",
                            test_idx=ev_full,
                        )
                        split_rows.append(
                            {
                                "combination": comb,
                                "cal": job["cal"],
                                "split": s,
                                "weight": wname,
                                "method": meth,
                                "alpha": a,
                                "r_hat": r,
                                "set_size": c,
                                "n_eval": int(ev_full.size),
                            }
                        )
                        acc.setdefault((wname, meth, a), []).append((r, c))
        for (wname, meth, a), vals in acc.items():
            rs = np.array([v[0] for v in vals], dtype=np.float64)
            cs = np.array([v[1] for v in vals], dtype=np.float64)
            split_sum.append(
                {
                    "combination": comb,
                    "cal": job["cal"],
                    "weight": wname,
                    "method": meth,
                    "alpha": a,
                    "n_splits": args.n_splits,
                    "r_mean": float(rs.mean()),
                    "r_sd": float(rs.std(ddof=1)),
                    "c_mean": float(cs.mean()),
                    "c_sd": float(cs.std(ddof=1)),
                    "frac_over_alpha": float((rs > a).mean()),
                }
            )
            print(
                "  split %s %s a=%.2f  R=%.4f +/- %.4f  |C|=%.1f +/- %.1f  "
                "P(R>a)=%.2f"
                % (
                    wname,
                    meth,
                    a,
                    rs.mean(),
                    rs.std(ddof=1),
                    cs.mean(),
                    cs.std(ddof=1),
                    (rs > a).mean(),
                ),
                flush=True,
            )

    if not args.skip_pac:
        print("\n=== Bernstein/APUB calibration resamples ===", flush=True)
        for job in jobs:
            comb = job["comb"]
            pac_names = CORE_WEIGHTS
            acc: dict[tuple[str, str, str, float], list[float]] = {}
            for d in range(args.n_pac):
                rng = np.random.default_rng(args.seed + 5000 + d)
                idx = _calib_cluster_idx(job["calib_ids"], rng)
                top1 = matched_gallery_top1(
                    job["calib_sims"][idx],
                    [str(x) for x in job["calib_ids"][idx]],
                    [str(x) for x in job["cg_ids"]],
                    n_gallery_ids=int(job["n_ids_test"]),
                    seed=int(args.seed + 8000 + d),
                )
                s_base = job["s_base"][idx]
                phi_m = np.clip(1.0 - top1, 0.0, 1.0) + BETA
                s_ad = s_base / phi_m
                t_cal = top1
                wmap = _build_1d_weights(t_cal, job["t_te"], pac_names)
                print(f"  {comb} draw {d + 1}/{args.n_pac}  n={idx.size}", flush=True)
                for wname, w in wmap.items():
                    for meth, scores, adaptive in (
                        ("A", s_base, False),
                        ("B", s_ad, True),
                    ):
                        unique_tau, _, apub = wacr.apub_bounds_on_grid(
                            scores, w, delta=DELTA, seed=args.seed + d
                        )
                        lookup = (unique_tau, apub)
                        for a in ALPHAS:
                            for bound in ("bernstein", "apub"):
                                _tau, r, c, _m, _s = _eval(
                                    wacr,
                                    job,
                                    scores,
                                    w,
                                    a,
                                    adaptive,
                                    bound=bound,
                                    apub_lookup=lookup if bound == "apub" else None,
                                )
                                pac_rows.append(
                                    {
                                        "combination": comb,
                                        "cal": job["cal"],
                                        "draw": d,
                                        "weight": wname,
                                        "method": meth,
                                        "alpha": a,
                                        "bound": bound,
                                        "r_hat": r,
                                        "set_size": c,
                                        "over": int(r > a),
                                    }
                                )
                                acc.setdefault((wname, meth, bound, a), []).append(r)
            for (wname, meth, bound, a), rs in acc.items():
                arr = np.asarray(rs, dtype=np.float64)
                rec = {
                    "combination": comb,
                    "cal": job["cal"],
                    "weight": wname,
                    "method": meth,
                    "bound": bound,
                    "alpha": a,
                    "n_pac": args.n_pac,
                    "frac_over": float((arr > a).mean()),
                    "r_mean": float(arr.mean()),
                    "r_sd": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
                }
                pac_sum.append(rec)
                print(
                    "  PAC %s %s %s a=%.2f  P(R>a)=%.3f  R=%.4f +/- %.4f"
                    % (
                        wname,
                        meth,
                        bound,
                        a,
                        rec["frac_over"],
                        rec["r_mean"],
                        rec["r_sd"],
                    ),
                    flush=True,
                )

    _write_csv(FIG / "validity_kmm.csv", kmm_rows)
    _write_csv(FIG / "validity_fullset.csv", full_rows)
    _write_csv(FIG / "validity_paired.csv", pair_rows)
    _write_csv(FIG / "validity_splits.csv", split_rows)
    _write_csv(FIG / "validity_splits_summary.csv", split_sum)
    _write_csv(FIG / "validity_pac.csv", pac_rows)
    _write_csv(FIG / "validity_pac_summary.csv", pac_sum)

    print("\n=== full-set conformal (Method B, a=0.10) ===", flush=True)
    print(
        f"{'comb':<16} {'weight':<14} {'R':>7} {'|C|':>7} {'R-a':>8} "
        f"{'strat CI':>18} {'ok':>4}",
        flush=True,
    )
    for row in full_rows:
        if row["method"] != "B" or abs(row["alpha"] - 0.10) > 1e-12:
            continue
        print(
            "%-16s %-14s %7.4f %7.1f %8.4f  [%6.3f, %6.3f]  %s"
            % (
                row["combination"],
                row["weight"],
                row["r_hat"],
                row["set_size"],
                row["r_minus_a"],
                row["strat_lo"],
                row["strat_hi"],
                "OK" if row["controlled"] else "FAIL",
            ),
            flush=True,
        )


