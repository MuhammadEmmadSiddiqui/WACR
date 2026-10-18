"""WRCP (Ai & Ren) helpers copied from eval_wrcp_airen.py and eval_wrcp_fdiv.py."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from baselines.ds_cp import l2_rows
from wacr.boot import _first_cross
from wacr.eval_b import _eval_b_per_query, empty_pct
from wacr.weights import decode_paths

P_FLOOR = 1e-10
SEED = 42
N_GALLERY = 585
RHOS = np.unique(
    np.round(
        np.concatenate(
            [
                np.array([0.0]),
                np.linspace(0.0002, 0.010, 25),
                np.linspace(0.012, 0.040, 15),
                np.linspace(0.045, 0.120, 16),
                np.linspace(0.130, 0.300, 12),
                np.linspace(0.320, 0.550, 8),
            ]
        ),
        6,
    )
)


def f_kl(x: float) -> float:
    if x <= 0.0:
        return float("inf")
    return x * math.log(x)


def invg(r: float, rho: float, f=f_kl, eps: float = 1e-10) -> float:
    if r > 1.0:
        return 1.0
    left = r
    right = 1.0
    mid = 0.5 * (left + right)
    while right - left > eps:
        if mid <= 0.0 or mid >= 1.0:
            break
        ans = mid * f(r / mid) + (1.0 - mid) * f((1.0 - r) / (1.0 - mid))
        if ans <= rho:
            left = mid
        else:
            right = mid
        mid = 0.5 * (left + right)
    return float(mid)


def quantile_weighted(seq: np.ndarray, wseq: np.ndarray, beta: float) -> float:
    seq = np.asarray(seq, dtype=np.float64).reshape(-1)
    wseq = np.asarray(wseq, dtype=np.float64).reshape(-1)
    sorted_indices = np.argsort(seq)
    sorted_seq = seq[sorted_indices]
    sorted_wseq = wseq[sorted_indices]
    cumulative_weights = np.cumsum(sorted_wseq)
    total = float(cumulative_weights[-1])
    desired_index = int(np.searchsorted(cumulative_weights, beta * total))
    if desired_index >= sorted_seq.size:
        return float("inf")
    q = sorted_seq[desired_index]
    if not np.isfinite(q):
        return float("inf")
    return float(q.item() if hasattr(q, "item") else q)


class PerQueryWRCP:
    def __init__(self, scores: np.ndarray, w_cal: np.ndarray):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        w_cal = np.asarray(w_cal, dtype=np.float64).reshape(-1)
        order = np.argsort(scores, kind="mergesort")
        self.s = scores[order]
        self.csum = np.cumsum(w_cal[order])
        self.w_sum = float(w_cal.sum())
        self.w_cal = w_cal
        self.scores = scores

    def tau(self, beta: float, w_x: float) -> float:
        total = self.w_sum + float(w_x)
        if total <= 0.0 or not 0.0 < beta <= 1.0:
            return float("inf")
        target = beta * total
        if target > self.csum[-1] + 1e-15:
            return float("inf")
        k = int(np.searchsorted(self.csum, target, side="left"))
        if k >= self.s.size:
            return float("inf")
        return float(self.s[k])

    def tau_slow(self, beta: float, w_x: float) -> float:
        seq = np.concatenate([self.scores, np.array([np.inf])])
        wseq = np.concatenate([self.w_cal, np.array([float(w_x)])])
        wseq = wseq / wseq.sum()
        return quantile_weighted(seq, wseq, beta)


def make_xgb():
    clf = XGBClassifier()
    clf.set_params(random_state=SEED, n_jobs=1)
    return clf, {k: str(v) for k, v in clf.get_params().items()}


def density_w(clf, x: np.ndarray, p0: float, p1: float) -> np.ndarray:
    p = clf.predict_proba(np.asarray(x, dtype=np.float64))
    p = np.maximum(p, P_FLOOR)
    return (p0 / p1) * (p[:, 1] / p[:, 0])


def fit_domain_clf(x_src: np.ndarray, x_tgt: np.ndarray, kind: str):
    x_src = l2_rows(x_src)
    x_tgt = l2_rows(x_tgt)
    p0 = float(x_src.shape[0])
    p1 = float(x_tgt.shape[0])
    y = np.concatenate(
        [np.zeros(x_src.shape[0], dtype=np.int32), np.ones(x_tgt.shape[0], dtype=np.int32)]
    )
    x = np.vstack([x_src, x_tgt])
    clf, raw = make_xgb()
    clf.fit(x, y)
    return clf, p0, p1, raw


def identity_folds(ids: np.ndarray, seed: int = SEED) -> np.ndarray:
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    mid = int(perm.size // 2)
    fold_of = {u: 0 if i < mid else 1 for i, u in enumerate(perm)}
    return np.array([fold_of[i] for i in ids], dtype=np.int32)


def two_fold_weights(
    x_src: np.ndarray,
    x_cal: np.ndarray,
    x_te: np.ndarray,
    fold: np.ndarray,
    kind: str,
) -> dict:
    n_te = int(x_te.shape[0])
    n_cal = int(x_cal.shape[0])
    w_cal_by_fold = np.zeros((2, n_cal), dtype=np.float64)
    w_te = np.zeros(n_te, dtype=np.float64)
    auc = []
    p_atom = np.zeros(n_te, dtype=np.float64)
    params = None
    for k in (0, 1):
        fit_te = x_te[fold != k]
        eval_ix = np.flatnonzero(fold == k)
        if fit_te.shape[0] < 2 or eval_ix.size == 0:
            raise RuntimeError(f"fold {k} too small")
        clf, p0, p1, raw = fit_domain_clf(x_src, fit_te, kind)
        params = raw
        w_cal = density_w(clf, x_cal, p0, p1)
        w_ev = density_w(clf, x_te[eval_ix], p0, p1)
        w_cal_by_fold[k] = w_cal
        w_te[eval_ix] = w_ev
        p_atom[eval_ix] = w_ev / (w_cal.sum() + w_ev)
        y = np.concatenate(
            [np.zeros(x_cal.shape[0], dtype=np.int32), np.ones(eval_ix.size, dtype=np.int32)]
        )
        p = np.concatenate(
            [
                clf.predict_proba(l2_rows(x_cal))[:, 1],
                clf.predict_proba(l2_rows(x_te[eval_ix]))[:, 1],
            ]
        )
        auc.append(float(roc_auc_score(y, p)))
    return {
        "w_cal_by_fold": w_cal_by_fold,
        "w_te": w_te,
        "p_atom": p_atom,
        "auc": auc,
        "params": params,
        "n_src": int(x_src.shape[0]),
        "n_cal": n_cal,
    }


def tau_vector(job, scores, w_cal_by_fold, w_te, fold, keep, beta: float) -> np.ndarray:
    n_test = int(job["test_sims"].shape[0])
    tau_q = np.full(n_test, np.inf, dtype=np.float64)
    engines = [PerQueryWRCP(scores, w_cal_by_fold[k]) for k in (0, 1)]
    for j, row in enumerate(keep):
        k = int(fold[j])
        tau_q[int(row)] = engines[k].tau(beta, float(w_te[j]))
    return tau_q


def align_fe_rows(z_paths: list[str], fe: np.ndarray, expected: list[str]) -> np.ndarray:
    idx = {p: i for i, p in enumerate(z_paths)}
    if all(p in idx for p in expected):
        return np.stack([fe[idx[p]] for p in expected], axis=0)
    by_name = {}
    for i, p in enumerate(z_paths):
        by_name[Path(str(p).replace("\\", "/")).name] = i
    return np.stack([fe[by_name[Path(str(p).replace("\\", "/")).name]] for p in expected], axis=0)


def load_pool_dinov2(repo: Path, cal: str) -> tuple[np.ndarray, list[str]]:
    path = repo / "cache" / "dinov2" / f"pool_{cal}_dinov2.npz"
    z = np.load(path, allow_pickle=True)
    fe = np.asarray(z["fe"], dtype=np.float64)
    paths = decode_paths(z["paths"])
    return fe, paths


def load_scface_dinov2(repo: Path, cal: str, expected_paths: list[str]) -> np.ndarray:
    path = repo / "cache" / "dinov2" / f"{cal}_SCFace_Queries_dinov2.npz"
    z = np.load(path, allow_pickle=True)
    fe = np.asarray(z["fe"], dtype=np.float64)
    cached = decode_paths(z["paths"])
    return align_fe_rows(cached, fe, expected_paths)
