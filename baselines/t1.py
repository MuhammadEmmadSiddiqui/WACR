"""T1 Method-B rows at alpha=0.10. Copied functions; one draw each."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from baselines.ds_cp import (
    align_fe,
    eval_tau as ds_eval_tau,
    fit_weights,
    l2_rows,
    split_cp_threshold,
    weighted_quantile,
)
from baselines.ltt_core import DELTA as LTT_DELTA
from baselines.ltt_core import KMM_K, median_pairwise_sigma, run_one, topk_desc
from baselines.pac_core import (
    DELTA,
    SEED,
    heuristic_iw,
    kmm_weights_qp,
    ps_w_tau,
    stats_from_tau,
    theorem5_bounds,
    train_domain_classifier,
)
from baselines.wrcp_core import (
    RHOS,
    SEED as WRCP_SEED,
    identity_folds,
    invg,
    load_pool_dinov2,
    load_scface_dinov2,
    tau_vector,
    two_fold_weights,
)
from wacr.boot import _first_cross
from wacr.eval_b import _eval_b_per_query, empty_pct
from wacr.embeddings import load_bundle
from wacr.rafr import k_cut, matched_rank_and_t, test_ranks
from wacr.protocol import BETA

A10 = 0.10
N_GALLERY = 585
N_GALLERY_IDS = 65
GMATCH_SEED = 0
M_MAIN = 20
DELTA_KAPPA = 0.5
REPO = Path(__file__).resolve().parents[1]


def q_strict(s: np.ndarray, c: float) -> float:
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    n = int(s.size)
    c = float(c)
    if n == 0:
        return float("nan")
    if c <= 0.0:
        return float("-inf")
    if c > 1.0:
        return float("inf")
    order = np.sort(s, kind="mergesort")
    k = int(math.ceil(n * c - 1e-15))
    if k <= 0:
        return float("-inf")
    if k > n:
        return float("inf")
    return float(order[k - 1])


def qtc_betas(s_p: np.ndarray, s_q: np.ndarray, alpha: float) -> dict:
    s_p = np.asarray(s_p, dtype=np.float64).reshape(-1)
    s_q = np.asarray(s_q, dtype=np.float64).reshape(-1)
    q_q = q_strict(s_q, alpha)
    q_p = q_strict(s_p, 1.0 - alpha)
    b_t = float(np.mean(s_p < q_q)) if s_p.size else float("nan")
    b_s = 1.0 - float(np.mean(s_q < q_p)) if s_q.size else float("nan")
    return {
        "beta_t": b_t,
        "beta_s": b_s,
        "beta_qtc": float(min(b_t, b_s)),
        "q_q": q_q,
        "q_p": q_p,
    }


def split_cp_k_tau(scores: np.ndarray, beta: float) -> tuple[int, float]:
    s = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1), kind="mergesort")
    n = int(s.size)
    b = float(beta)
    if not np.isfinite(b):
        b = 0.0
    b = min(max(b, 0.0), 1.0)
    k = int(math.ceil((1.0 - b) * (n + 1)))
    if k < 1:
        return k, float("-inf")
    if k > n:
        return k, float("inf")
    return k, float(s[k - 1])


def eval_const_tau(job: dict, tau: float):
    tau_q = np.full(int(job["test_sims"].shape[0]), float(tau), dtype=np.float64)
    return _eval_b_per_query(job, tau_q)


def run_qtc(job: dict, alpha: float = A10):
    scores_b = np.asarray(job["s_g"], dtype=np.float64)
    t_p = np.asarray(job["t_cal"], dtype=np.float64)
    t_q = np.asarray(job["t_te"], dtype=np.float64)[job["test_valid"]]
    betas = qtc_betas(t_p, t_q, alpha)
    _k, tau = split_cp_k_tau(scores_b, float(betas["beta_qtc"]))
    r, c, miss, sizes = eval_const_tau(job, tau)
    return r, c, empty_pct(sizes), miss


def equal_width_edges(t: np.ndarray, m: int) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    lo = float(t.min())
    hi = float(t.max())
    if hi <= lo:
        hi = lo + 1e-12
    return np.linspace(lo, hi, int(m) + 1)


def pooled_edges(t_cal: np.ndarray, t_te: np.ndarray, m: int) -> np.ndarray:
    return equal_width_edges(np.concatenate([t_cal.reshape(-1), t_te.reshape(-1)]), m)


def digitize_t(t: np.ndarray, edges: np.ndarray) -> np.ndarray:
    m = int(edges.size - 1)
    idx = np.digitize(np.asarray(t, dtype=np.float64).reshape(-1), edges) - 1
    return np.clip(idx, 0, m - 1).astype(np.int64)


def mondrian_tau(scores: np.ndarray, bins: np.ndarray, m: int, alpha: float) -> np.ndarray:
    tau = np.full(int(m), np.inf, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    bins = np.asarray(bins, dtype=np.int64).reshape(-1)
    for g in range(int(m)):
        s = scores[bins == g]
        tau[g] = split_cp_threshold(s, alpha)
    return tau


def tau_from_bins(bins_te: np.ndarray, tau_bin: np.ndarray) -> np.ndarray:
    return np.asarray(tau_bin, dtype=np.float64)[np.asarray(bins_te, dtype=np.int64)]


def run_condcp(job: dict, alpha: float = A10):
    scores = np.asarray(job["s_g"], dtype=np.float64)
    t_cal = np.asarray(job["t_cal"], dtype=np.float64)
    t_te = np.asarray(job["t_te"], dtype=np.float64)
    edges = pooled_edges(t_cal, t_te, M_MAIN)
    bins_cal = digitize_t(t_cal, edges)
    bins_te = digitize_t(t_te, edges)
    tau_m = mondrian_tau(scores, bins_cal, int(edges.size - 1), alpha)
    tau_q = tau_from_bins(bins_te, tau_m)
    r, c, miss, sizes = _eval_b_per_query(job, tau_q)
    return r, c, empty_pct(sizes), miss


def calibrate_kappa_hoeffding(ranks, n_gal, phi, alpha, n_te, mode: str):
    n = int(ranks.size)
    kappa_max = float(n_te)
    grid_size = int(math.ceil((kappa_max - 1.0) / DELTA_KAPPA)) + 1
    pen = math.sqrt(math.log(1.0 / DELTA) / (2.0 * n))
    for k in range(grid_size):
        kappa = 1.0 + k * DELTA_KAPPA
        k_cal, _ = k_cut(kappa, phi, n_gal, n_te, mode)
        rho = float((ranks > k_cal).mean())
        if rho + pen <= alpha + 1e-12:
            return float(kappa), rho, rho + pen, pen, grid_size
    return None, float("nan"), float("nan"), pen, grid_size


def run_rafr(job: dict, alpha: float = A10):
    t, rank, n_gal = matched_rank_and_t(
        job["calib_sims"],
        job["calib_ids"],
        job["cg_ids"],
        n_gallery_ids=int(job["n_ids_test"]),
        seed=GMATCH_SEED,
    )
    phi_cal = np.clip(1.0 - t, 0.0, 1.0) + BETA
    phi_te = np.clip(1.0 - job["t_te"], 0.0, 1.0) + BETA
    te_rank = test_ranks(job["test_sims"], job["true_test"])
    valid = te_rank > 0
    n_te = int(job["test_sims"].shape[1])
    khat, rho, u, pen, gsz = calibrate_kappa_hoeffding(
        rank, n_gal, phi_cal, alpha, n_te, "percentile"
    )
    if khat is None:
        miss = np.ones(int(valid.sum()), dtype=np.float64)
        return 1.0, float(n_te), 0.0, miss
    _, k_final = k_cut(khat, phi_te, np.full(phi_te.shape, n_te), n_te, "percentile")
    k_v = k_final[valid]
    miss = (te_rank[valid] > k_v).astype(np.float64)
    return float(miss.mean()), float(k_v.mean()), 0.0, miss


def fit_paccs(wacr, job: dict) -> None:
    if job.get("w_paccs_lo") is not None:
        return
    if job.get("from_scores"):
        raise RuntimeError("PAC-CS weights missing from score cache")
    scores = np.asarray(job["s_g"], dtype=np.float64)
    ada_cal = np.asarray(job["calib_emb"], dtype=np.float64)
    ada_te = np.asarray(job["test_emb"], dtype=np.float64)[job["test_valid"]]
    src_ids = np.asarray(job["calib_ids"])
    tgt_ids = np.asarray(job["test_ids"])
    print(f"  PAC-CS {job['cal']} train MidFNN on AdaFace ...", flush=True)
    p_src, p_tgt, clf_meta = train_domain_classifier(ada_cal, ada_te, src_ids, tgt_ids, SEED)
    rng_iw = np.random.default_rng(SEED + 11)
    w_hat_src = heuristic_iw(p_src, rng_iw)
    w_hat_tgt = heuristic_iw(p_tgt, rng_iw)
    w_lo, w_hi, _edges, bmeta = theorem5_bounds(w_hat_src, w_hat_tgt, delta_w=DELTA / 2.0)
    print(
        f"    AUC={clf_meta['auc_holdout']:.4f} device={clf_meta['device']} "
        f"b={bmeta['b']:.4g}",
        flush=True,
    )
    job["w_paccs"] = w_hat_src
    job["w_paccs_lo"] = w_lo
    job["w_paccs_hi"] = w_hi
    job["u_paccs"] = np.random.default_rng(SEED).random(scores.size)


def run_paccs(wacr, job: dict, alpha: float = A10):
    fit_paccs(wacr, job)
    scores = np.asarray(job["s_g"], dtype=np.float64)
    tau, n_acc, k, ucb, b = ps_w_tau(
        scores, job["w_paccs_lo"], job["w_paccs_hi"], job["u_paccs"], alpha, DELTA / 2.0
    )
    st = stats_from_tau(wacr, job, tau)
    return st["r_hat"], st["set_size"], st["empty_pct"], st["miss"]


def fit_ltt(wacr, job: dict) -> None:
    if job.get("w_ltt") is not None:
        return
    if job.get("from_scores"):
        raise RuntimeError("LTT-IW weights missing from score cache")
    x_src = topk_desc(job["calib_sims"], KMM_K)
    x_tgt = topk_desc(job["test_sims"][job["test_valid"]], KMM_K)
    print(f"  LTT-IW {job['cal']} KMM QP n={x_src.shape[0]} ...", flush=True)
    w_kmm, meta = kmm_weights_qp(
        x_src, x_tgt, seed=SEED, median_pairwise_sigma=median_pairwise_sigma
    )
    print(f"    QP status={meta['status']} ok={meta['ok']} {meta['seconds']:.1f}s", flush=True)
    job["w_ltt"] = w_kmm
    job["w_ltt_ok"] = bool(int(meta["ok"])) and np.all(np.isfinite(w_kmm))


def run_ltt(wacr, job: dict, alpha: float = A10):
    fit_ltt(wacr, job)
    scores = np.asarray(job["s_g"], dtype=np.float64)
    w_kmm = job["w_ltt"]
    qp_ok = job.get("w_ltt_ok", np.all(np.isfinite(w_kmm)))
    rng0 = np.random.default_rng(SEED)
    if qp_ok:
        st, diag = run_one(wacr, job, scores, w_kmm, alpha, LTT_DELTA, "wsr", rng0)
    else:
        st = stats_from_tau(wacr, job, float("inf"))
    return st["r_hat"], st["set_size"], st["empty_pct"], st["miss"]


def fit_wrcp(wacr, job: dict) -> None:
    if job.get("w_wrcp_cal") is not None:
        return
    if job.get("from_scores"):
        raise RuntimeError("WRCP weights missing from score cache")
    cal = job["cal"]
    wx_path = job["cache"] / f"{cal}_Queries_wx.npz"
    x_cal_dino = l2_rows(align_fe(wx_path, job["calib_paths"]))
    x_src, _src_paths = load_pool_dinov2(REPO, cal)
    x_src = l2_rows(x_src)
    test = load_bundle(job["cache"] / "SCFace_Queries_mcd.npz", name="SCFace_Queries")
    true_all = job["true_test"]
    keep = np.flatnonzero(true_all >= 0)
    paths_kept = [test.paths[int(i)] for i in keep]
    x_te_dino = l2_rows(load_scface_dinov2(REPO, cal, paths_kept))
    te_ids = np.asarray(test.identities)[keep]
    fold = identity_folds(te_ids, WRCP_SEED)
    print(
        f"  WRCP {cal} pool n={x_src.shape[0]} cal={x_cal_dino.shape[0]} te={x_te_dino.shape[0]}",
        flush=True,
    )
    pack = two_fold_weights(x_src, x_cal_dino, x_te_dino, fold, "xgb")
    print(f"    AUC mean={float(np.mean(pack['auc'])):.4f}", flush=True)
    job["w_wrcp_cal"] = pack["w_cal_by_fold"]
    job["w_wrcp_te"] = pack["w_te"]
    job["w_wrcp_fold"] = fold
    job["w_wrcp_keep"] = keep


def run_wrcp(wacr, job: dict, alpha: float = A10):
    fit_wrcp(wacr, job)
    scores = np.asarray(job["s_g"], dtype=np.float64)
    keep = np.asarray(job["w_wrcp_keep"], dtype=np.int64)
    fold = np.asarray(job["w_wrcp_fold"])
    xs, rhats, csizes, empty, misses = [], [], [], [], []
    for rho in RHOS:
        beta = invg(1.0 - alpha, float(rho))
        tau_q = tau_vector(
            job, scores, job["w_wrcp_cal"], job["w_wrcp_te"], fold, keep, beta
        )
        r, c, miss, sizes = _eval_b_per_query(job, tau_q)
        xs.append(float(rho))
        rhats.append(r)
        csizes.append(c)
        empty.append(empty_pct(sizes))
        misses.append(miss)
    star = _first_cross(np.array(xs), np.array(rhats), alpha)
    if not np.isfinite(star):
        return float(rhats[0]), float(csizes[0]), float(empty[0]), misses[0]
    at = min(range(len(xs)), key=lambda i: abs(xs[i] - star))
    return float(rhats[at]), float(csizes[at]), float(empty[at]), misses[at]


def fit_dscp(wacr, job: dict) -> None:
    if job.get("w_dscp") is not None:
        return
    if job.get("from_scores"):
        raise RuntimeError("DS-CP weights missing from score cache")
    cal = job["cal"]
    wx_path = job["cache"] / f"{cal}_Queries_wx.npz"
    fe_dino_cal = align_fe(wx_path, job["calib_paths"])
    test = load_bundle(job["cache"] / "SCFace_Queries_mcd.npz", name="SCFace_Queries")
    keep = np.flatnonzero(job["true_test"] >= 0)
    paths_kept = [test.paths[int(i)] for i in keep]
    fe_dino_te = load_scface_dinov2(REPO, cal, paths_kept)
    print(f"  DS-CP {cal} XGB DINOv2 ...", flush=True)
    job["w_dscp"] = fit_weights(fe_dino_cal, fe_dino_te, "XGBoost")


def run_dscp(wacr, job: dict, alpha: float = A10):
    fit_dscp(wacr, job)
    scores = job["s_g"]
    tau = weighted_quantile(scores, job["w_dscp"], alpha, gamma=1.0)
    r, c = ds_eval_tau(wacr, job, tau, 0)
    res = wacr.evaluate_adaptive(
        job["test_sims"], tau, job["phi_test"], job["true_test"], min_set_size=0
    )
    valid = job["true_test"] >= 0
    miss = (~res.contains_true[valid]).astype(np.float64)
    return float(r), float(c), empty_pct(res.set_sizes[valid]), miss


def fit_baseline_weights(wacr, jobs) -> None:
    for job in jobs:
        print(f"=== fit T1 baseline weights {job['cal']} ===", flush=True)
        fit_dscp(wacr, job)
        fit_wrcp(wacr, job)
        fit_paccs(wacr, job)
        fit_ltt(wacr, job)


def run_copied_baselines(wacr, jobs) -> dict:
    """Return {(cal, method): (R, |C|, empty_pct)} for T1 copied methods."""
    fit_baseline_weights(wacr, jobs)
    out = {}
    for job in jobs:
        cal = job["cal"]
        print(f"=== T1 copied baselines {cal} ===", flush=True)
        out[(cal, "QTC")] = run_qtc(job)[:3]
        print(f"  QTC R={out[(cal, 'QTC')][0]:.6f} |C|={out[(cal, 'QTC')][1]:.4f}", flush=True)
        out[(cal, "Conditional CP")] = run_condcp(job)[:3]
        print(
            f"  CondCP R={out[(cal, 'Conditional CP')][0]:.6f} "
            f"|C|={out[(cal, 'Conditional CP')][1]:.4f}",
            flush=True,
        )
        out[(cal, "RA-FR")] = run_rafr(job)[:3]
        print(f"  RA-FR R={out[(cal, 'RA-FR')][0]:.6f} |C|={out[(cal, 'RA-FR')][1]:.4f}", flush=True)
        out[(cal, "DS-CP")] = run_dscp(wacr, job)[:3]
        print(f"  DS-CP R={out[(cal, 'DS-CP')][0]:.6f} |C|={out[(cal, 'DS-CP')][1]:.4f}", flush=True)
        out[(cal, "WRCP")] = run_wrcp(wacr, job)[:3]
        print(f"  WRCP R={out[(cal, 'WRCP')][0]:.6f} |C|={out[(cal, 'WRCP')][1]:.4f}", flush=True)
        out[(cal, "PAC-CS")] = run_paccs(wacr, job)[:3]
        print(
            f"  PAC-CS R={out[(cal, 'PAC-CS')][0]:.6f} |C|={out[(cal, 'PAC-CS')][1]:.4f}",
            flush=True,
        )
        out[(cal, "LTT-IW")] = run_ltt(wacr, job)[:3]
        print(f"  LTT-IW R={out[(cal, 'LTT-IW')][0]:.6f} |C|={out[(cal, 'LTT-IW')][1]:.4f}", flush=True)
    return out
