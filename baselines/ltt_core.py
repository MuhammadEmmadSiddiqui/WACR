"""LTT-IW WSR / FST as copied from Outputs/eval_ltt_iw.py."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics.pairwise import euclidean_distances

from baselines.pac_core import N_GALLERY, stats_from_tau
from wacr.eval_b import _eval_b

N_PERM = 10
N_GRID = 1000
KMM_K = 10
DELTA = 0.1


def topk_desc(sims: np.ndarray, k: int = KMM_K) -> np.ndarray:
    sims = np.asarray(sims, dtype=np.float64)
    k = min(int(k), int(sims.shape[1]))
    part = np.partition(sims, sims.shape[1] - k, axis=1)[:, -k:]
    return np.sort(part, axis=1)[:, ::-1]


def median_pairwise_sigma(X: np.ndarray, n_sample: int = 2500, seed: int = 0) -> float:
    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0])
    m = min(int(n_sample), n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    d = euclidean_distances(X[idx], X[idx], squared=False)
    tri = d[np.triu_indices(m, k=1)]
    return float(np.median(tri)) if tri.size else 1.0


def bernstein_p_value(x: np.ndarray, mu0: float, M: float = 1.0) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n <= 1:
        return 1.0
    rhat = float(x.mean())
    gap = float(mu0) - rhat
    if gap <= 1e-15:
        return 1.0
    v = max(0.0, float(np.var(x, ddof=1)))
    a = 7.0 * float(M) / (3.0 * (n - 1))
    b = math.sqrt(2.0 * v / n)
    if a <= 1e-18:
        t = (gap / b) ** 2 if b > 1e-18 else float("inf")
    else:
        disc = b * b + 4.0 * a * gap
        u = (-b + math.sqrt(max(disc, 0.0))) / (2.0 * a)
        t = u * u
    return float(min(1.0, 2.0 * math.exp(-t)))


def hb_p_value(mean_l: float, n: int, mu0: float) -> float:
    if n <= 0:
        return 1.0
    gap = float(mu0) - float(mean_l)
    if gap <= 1e-15:
        return 1.0
    return float(min(1.0, math.exp(-2.0 * n * gap * gap)))


def wsr_log_capital_batch(X: np.ndarray, mu0: float, delta: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    p, n = X.shape
    i = np.arange(1, n + 1, dtype=np.float64)
    csum = np.cumsum(X, axis=1)
    muhat = (0.5 + csum) / (1.0 + i)
    var_c = np.cumsum((X - muhat) ** 2, axis=1)
    sigma2 = (0.25 + var_c) / (1.0 + i)
    sigma2_prev = np.empty_like(sigma2)
    sigma2_prev[:, 0] = 0.25
    sigma2_prev[:, 1:] = sigma2[:, :-1]
    sigma2_prev = np.maximum(sigma2_prev, 1e-12)
    nu = np.minimum(
        1.0,
        np.sqrt(2.0 * math.log(1.0 / float(delta)) / (float(n) * sigma2_prev)),
    )
    z = 1.0 - nu * (X - float(mu0))
    z = np.clip(z, 1e-15, None)
    return np.cumsum(np.log(z), axis=1)


def wsr_p_on_grid(
    scores: np.ndarray,
    wB: np.ndarray,
    taus: np.ndarray,
    mu0: float,
    delta: float,
    perms: np.ndarray,
) -> np.ndarray:
    L = (wB[None, :] * (scores[None, :] > taus[:, None])).astype(np.float64)
    x = L[:, perms]
    t, p, n = x.shape
    logK = wsr_log_capital_batch(x.reshape(t * p, n), mu0, delta)
    logKavg = logsumexp(logK.reshape(t, p, n), axis=1) - math.log(p)
    m = logKavg.max(axis=1)
    pval = np.ones(t, dtype=np.float64)
    pos = m > 0.0
    pval[pos] = np.minimum(1.0, np.exp(-m[pos]))
    return pval


def score_quantile_grid(scores: np.ndarray, n_grid: int = N_GRID) -> np.ndarray:
    q = np.linspace(1.0 / float(n_grid), 1.0, int(n_grid))
    g = np.quantile(np.asarray(scores, dtype=np.float64).reshape(-1), q)
    return np.sort(np.unique(np.asarray(g, dtype=np.float64)))[::-1]


def rescale_B(w: np.ndarray) -> float:
    return max(float(np.max(w)), 1e-12)


def fst_tau(
    scores: np.ndarray,
    w: np.ndarray,
    alpha: float,
    delta: float,
    kind: str,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> tuple[float, dict]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    n = int(scores.size)
    B = rescale_B(w)
    mu0 = float(alpha) / B
    wB = w / B
    taus = score_quantile_grid(scores, N_GRID)
    n_perm = max(1, int(n_perm))
    if n_perm <= 1:
        perms = np.arange(n, dtype=np.int64)[None, :]
    else:
        perms = np.stack([rng.permutation(n) for _ in range(n_perm)])

    last = None
    n_tested = 0
    n_reject = 0
    p_last_rej = None
    p_stop = None
    chunk = 40 if kind == "wsr" else taus.size
    stop = False
    for start in range(0, taus.size, chunk):
        sl = taus[start : start + chunk]
        if kind == "wsr":
            pvals = wsr_p_on_grid(scores, wB, sl, mu0, delta, perms)
        else:
            pvals = np.empty(sl.size, dtype=np.float64)
            for i, tau in enumerate(sl):
                L = wB * (scores > tau)
                if kind == "hb":
                    pvals[i] = hb_p_value(float(L.mean()), n, mu0)
                elif kind == "bernstein":
                    pvals[i] = bernstein_p_value(L, mu0)
                else:
                    raise ValueError(kind)
        for tau, pval in zip(sl, pvals):
            n_tested += 1
            if float(pval) <= float(delta) + 1e-15:
                n_reject += 1
                last = float(tau)
                p_last_rej = float(pval)
            else:
                p_stop = float(pval)
                stop = True
                break
        if stop:
            break
    tau_hat = float("inf") if last is None else float(last)
    diag = {
        "B": B,
        "mu0": mu0,
        "n_grid": int(taus.size),
        "n_tested": n_tested,
        "n_reject": n_reject,
        "p_last_rej": p_last_rej,
        "p_stop": p_stop,
        "tau": tau_hat,
        "kind": kind,
        "n_perm": int(n_perm) if kind == "wsr" else 0,
    }
    return tau_hat, diag


def eval_tau(wacr, job, tau: float):
    r, c, miss, sizes = _eval_b(wacr, job, tau)
    return float(r), float(c), miss, sizes


def run_one(wacr, job, scores, w, alpha, delta, kind, rng) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    tau, diag = fst_tau(scores, w, alpha, delta, kind, rng)
    r, c, miss, sizes = eval_tau(wacr, job, tau)
    st = stats_from_tau(wacr, job, tau)
    st["seconds"] = time.perf_counter() - t0
    st["miss"] = miss
    st["sizes"] = sizes
    st["r_hat"] = r
    st["set_size"] = c
    return st, diag
