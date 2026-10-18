#!/usr/bin/env python3
"""Weighted Adaptive Conformal Retrieval (WACR) under covariate shift.

Method A (baseline): non-conformity score = ``1 - cos_sim(query, gallery)``.
Method B (adaptive): non-conformity score = ``(1 - cos_sim) / Phi(x)``, where
``Phi(x)`` is an uncertainty adapter (top-1 distance or MCD ``fu``, plus ``beta``).

Calibration uses importance weights on source-domain queries (e.g. TinyFace)
to target the test domain (e.g. SCFace). Weight caches live under
``cache/<combination>/`` and are produced by ``Inference/`` scripts
(domain classifier ``*_wx.npz``, KMM ``*_kmm_*.npz``, KL ``*_kl.npz``, etc.);
each file must provide keys ``w`` and ``paths`` aligned to calibration queries.

Example::

    python "WACR/Weighted Adaptive Conformal Retrieval.py" \\
        --pair tinyface_scface \\
        --weights-file TinyFace_Queries_wx.npz \\
        --alpha 0.1 --beta 0.5 --uncertainty-method top1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

WACR_DIR = Path(__file__).resolve().parent
REPO_ROOT = WACR_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wacr.embeddings import EmbeddingBundle, load_bundle
from wacr.paths import CACHE_DIR, resolve_path
from wacr.similarity import cosine_similarity_matrix, l2_normalize
from wacr.uncertainty import UncertaintyAdapter, top1_fu_from_similarities
from wacr.weights import (
    effective_sample_size,
    importance_weight_cap,
    load_importance_weights,
)

DEFAULT_WEIGHTS_FILE = "TinyFace_Queries_wx.npz"


# ---------------------------------------------------------------------------
# Ground-truth gallery indices
# ---------------------------------------------------------------------------

def build_gallery_identity_index(gallery: EmbeddingBundle) -> Dict[str, List[int]]:
    """Map identity string -> list of gallery row indices for that identity."""
    idx: Dict[str, List[int]] = {}
    for i, ident in enumerate(gallery.identities):
        idx.setdefault(ident, []).append(i)
    return idx


def true_match_indices(
    queries: EmbeddingBundle,
    gallery: EmbeddingBundle,
    *,
    similarity_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Index of the ground-truth gallery image for each query (``-1`` if absent).

    When the gallery has multiple entries for an identity, the entry with the
    highest cosine similarity to the query is selected as the canonical match.
    """
    if similarity_fn is None:
        similarity_fn = cosine_similarity_matrix
    by_id = build_gallery_identity_index(gallery)
    sims = similarity_fn(queries.embeddings, gallery.embeddings)
    out = np.full(len(queries.paths), -1, dtype=np.int64)
    for qi, ident in enumerate(queries.identities):
        cands = by_id.get(ident)
        if not cands:
            continue
        cands_arr = np.asarray(cands, dtype=np.int64)
        out[qi] = int(cands_arr[int(np.argmax(sims[qi, cands_arr]))])
    return out


# ---------------------------------------------------------------------------
# Non-conformity scores and weighted conformal threshold
# ---------------------------------------------------------------------------

def calibration_nonconformity(
    queries: EmbeddingBundle, gallery: EmbeddingBundle, true_idx: np.ndarray
) -> np.ndarray:
    """``1 - cos_sim(query, true_gallery_match)`` for every calibration query."""
    if not np.all(true_idx >= 0):
        missing = int((true_idx < 0).sum())
        raise RuntimeError(f"{missing} calibration queries lack a gallery match")
    q = l2_normalize(queries.embeddings)
    g = l2_normalize(gallery.embeddings[true_idx])
    sims = np.einsum("ij,ij->i", q, g)
    return (1.0 - sims).astype(np.float64)


def weighted_conformal_threshold(
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    w_test: float | None = None,
) -> float:
    """Weighted split-conformal threshold under covariate shift (Tibshirani 2019).

    Returns the smallest non-conformity score whose cumulative weight reaches the
    ``1 - alpha`` level of the weighted empirical distribution that *includes an
    atom for the (unknown-weight) test point* at ``+inf``. The normalization
    denominator is ``sum(w) + w_test``.

    ``w_test`` must upper-bound the unknown test weight ``w(x_test)`` for the
    coverage theorem to apply; pass the finalization cap ``B`` (see
    ``utils.weights.importance_weight_cap``). When ``None``, the largest
    calibration weight is used (legacy behaviour; valid only if
    ``w(x_test) <= max_i w_i``).

    When ``1 - alpha`` exceeds the maximum mass achievable on calibration scores
    (``sum(w) / (sum(w) + w_test)``), the required level falls on the ``+inf`` atom
    and this function returns ``+inf`` (accept the whole gallery).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if scores.shape != weights.shape:
        raise ValueError("scores and weights must have the same length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if np.any(weights < 0):
        raise ValueError("importance weights must be non-negative")

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    w_sorted = weights[order]

    w_sum = float(w_sorted.sum())
    if w_sum <= 0.0:
        raise ValueError("sum of importance weights is non-positive; cannot normalize")

    # Test-point atom at +inf: bound the unknown test weight (Tibshirani 2019).
    if w_test is None:
        w_test = float(w_sorted.max())
    w_test = float(w_test)
    if w_test < float(w_sorted.max()):
        raise ValueError(
            f"w_test={w_test} is below max calibration weight {float(w_sorted.max())}"
        )
    cum = np.cumsum(w_sorted) / (w_sum + w_test)

    mask = cum >= (1.0 - alpha)
    if not np.any(mask):
        # Required mass lands on the +inf test atom: accept the whole gallery.
        return float("inf")
    return float(s_sorted[int(np.argmax(mask))])


def robust_weighted_conformal_threshold(
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    w_test: float | None = None,
) -> float:
    """Jin, Ren and Candes (PNAS 2023) Algorithm 1 on residual P(S|t) shift.

    ``weights`` are the identifiable t-weights (hist ratio, already finalized).
    The unknown likelihood ratio P_tgt(s|t)/P_src(s|t) is assumed to lie in
    ``[1/gamma, gamma]``. Then

        ell_i = w_i / gamma,   u_i = w_i * gamma,   u_{n+1} = w_test * gamma

    and the threshold is V_{[k*]} with
    k* = min {k : F(k) >= 1-alpha} and

        F(k) = sum_{i<=k} ell_[i] / (sum_{i<=k} ell_[i] + sum_{i>k} u_[i] + u_{n+1}).

    ``gamma=1`` recovers ``weighted_conformal_threshold``. ``w_test`` is the
    Tibshirani cap B on the identifiable test weight (default: max calibration
    weight). Coverage is conditional on the user's Gamma, not estimated from
    the target labels.

    If ``F(n) < 1-alpha`` the test atom carries the remaining mass and this
    function returns ``+inf`` (whole gallery), matching Tibshirani. Returning
    the largest finite calibration score would under-cover in that regime.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if scores.shape != weights.shape:
        raise ValueError("scores and weights must have the same length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    if np.any(weights < 0):
        raise ValueError("importance weights must be non-negative")

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    w_sorted = weights[order]
    w_sum = float(w_sorted.sum())
    if w_sum <= 0.0:
        raise ValueError("sum of importance weights is non-positive; cannot normalize")

    if w_test is None:
        w_test = float(w_sorted.max())
    w_test = float(w_test)
    if w_test + 1e-12 < float(w_sorted.max()):
        raise ValueError(
            f"w_test={w_test} is below max calibration weight {float(w_sorted.max())}"
        )

    g = float(gamma)
    ell = w_sorted / g
    u = w_sorted * g
    u_test = w_test * g
    c_ell = np.cumsum(ell)
    c_u = np.cumsum(u)
    sum_u = float(u.sum())
    den = c_ell + (sum_u - c_u) + u_test
    f = c_ell / np.maximum(den, 1e-30)
    mask = f >= (1.0 - alpha)
    if not np.any(mask):
        return float("inf")
    return float(s_sorted[int(np.argmax(mask))])


B_LOSS = 1.0  # binary miss loss for concentration bounds
APUB_N_BOOT = 3000  # Liu, Gao & Hu (ICML 2026) default M


def apub_from_bootstrap_means(boot_means: np.ndarray, alpha: float) -> np.ndarray:
    """Average-percentile UCB on a mean (Liu, Gao & Hu, ICML 2026).

    Definition 2.1: ``U_APUB^alpha = (1/alpha) int_0^alpha U_Efron^tau d tau``,
    where Efron's UCB is the left ``(1-tau)``-quantile of the bootstrap sample
    means ``mu*_m = N^{-1} sum_n F(zeta_{m,n})``. On the empirical bootstrap
    measure this is also Proposition 2.2,
    ``inf_t { t + alpha^{-1} E[(mu* - t)_+] }`` (Rockafellar-Uryasev / CVaR).

    ``boot_means`` has shape ``(M,)`` or ``(M, G)``. Returns shape ``()`` or
    ``(G,)``. ``alpha`` is the paper's APUB level (nominal coverage ``1-alpha``).
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("APUB alpha must be in (0, 1]")
    mu = np.asarray(boot_means, dtype=np.float64)
    squeeze = False
    if mu.ndim == 1:
        mu = mu[:, None]
        squeeze = True
    elif mu.ndim != 2:
        raise ValueError("boot_means must be 1-D or 2-D")
    m = int(mu.shape[0])
    if m < 1:
        raise ValueError("need at least one bootstrap mean")
    xs = np.sort(mu, axis=0)
    # Order statistic xs[i] = X_{(i+1)} is Efron's UCB on the tau-interval
    #   (max(0, 1-(i+1)/M), min(alpha, 1-i/M)].
    i = np.arange(m, dtype=np.float64)
    tau_lo = np.maximum(0.0, 1.0 - (i + 1.0) / m)
    tau_hi = np.minimum(alpha, 1.0 - i / m)
    weights = np.maximum(tau_hi - tau_lo, 0.0)
    out = (weights @ xs) / alpha
    return out[0] if squeeze else out



def concentration_threshold(
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    delta: float,
    kind: str,
) -> Tuple[float, dict]:
    """Smallest tau with a Bonferroni concentration upper bound U(tau) <= alpha.

    ``kind`` is ``bernstein`` (Empirical Bernstein, same form as W-RCR) or
    ``hoeffding`` (one-sided Hoeffding on weighted binary losses).

    Candidate thresholds are the unique calibration scores. Weighted losses are
    ``w_tilde_i * 1{s_i > tau}`` with ``mean(w_tilde) = 1``, matching W-RCR.
    If no candidate satisfies ``U(tau) <= alpha``, return ``+inf`` (whole gallery).
    Returning the largest finite calibration score would under-cover.
    """
    if kind not in {"bernstein", "hoeffding"}:
        raise ValueError(f"kind must be bernstein or hoeffding; got {kind!r}")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if scores.shape != weights.shape:
        raise ValueError("scores and weights must have the same length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if np.any(weights < 0):
        raise ValueError("importance weights must be non-negative")

    n = int(scores.size)
    if n < 2:
        raise ValueError("concentration bounds need at least 2 calibration scores")
    w_sum = float(weights.sum())
    if w_sum <= 0.0:
        raise ValueError("sum of importance weights is non-positive")

    w_tilde = (n / w_sum) * weights
    w_max = float(w_tilde.max())
    unique_tau = np.unique(scores)
    grid_size = int(unique_tau.size)
    log_two = math.log(2.0 * grid_size / delta)
    log_one = math.log(grid_size / delta)
    var_coef = 2.0 * log_two / n
    bias_term = 7.0 * w_max * B_LOSS * log_two / (3.0 * (n - 1))
    hoeff_pen = w_max * math.sqrt(log_one / (2.0 * n))

    chosen: dict | None = None
    last: dict | None = None
    for tau in unique_tau:
        losses = (scores > tau).astype(np.float64)
        wl = w_tilde * losses
        mu_hat = float(wl.mean())
        if kind == "hoeffding":
            penalty = hoeff_pen
        else:
            v_hat = max(0.0, float(np.var(wl, ddof=1)))
            penalty = math.sqrt(var_coef * v_hat) + bias_term
        u_hat = mu_hat + penalty
        last = {
            "tau": float(tau),
            "mu_hat": mu_hat,
            "penalty": penalty,
            "U": u_hat,
            "grid_size": grid_size,
        }
        if u_hat <= alpha:
            chosen = last
            break

    if chosen is None:
        assert last is not None
        last["feasible"] = False
        last["tau"] = float("inf")
        return float("inf"), last
    chosen["feasible"] = True
    return float(chosen["tau"]), chosen


def apub_bounds_on_grid(
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    delta: float,
    n_boot: int = APUB_N_BOOT,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """APUB (Def. 2.1) and point estimates on the unique-score grid.

    Cost at a candidate threshold is the weighted miss
    ``F_i(tau) = w_tilde_i 1{s_i > tau}`` with ``mean(w_tilde)=1``. Bootstrap
    sample means of ``F`` (multinomial SAM, same as Liu et al.) are converted
    to APUB by averaging Efron percentile UCBs from ``0`` to ``delta``, where
    ``delta`` is the paper's APUB level ``alpha`` (nominal ``1-delta``).
    Returns ``(unique_tau, mu_hat, apub)``. Non-increasing in ``tau``.
    Asymptotic, not PAC.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if scores.shape != weights.shape:
        raise ValueError("scores and weights must have the same length")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if np.any(weights < 0):
        raise ValueError("importance weights must be non-negative")

    n = int(scores.size)
    if n < 2:
        raise ValueError("APUB needs at least 2 calibration scores")
    w_sum = float(weights.sum())
    if w_sum <= 0.0:
        raise ValueError("sum of importance weights is non-positive")

    w_tilde = (n / w_sum) * weights
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    wt_sorted = w_tilde[order]

    # Bootstrap counts on the original indices, then align to ascending scores.
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    contrib = counts[:, order].astype(np.float64) * wt_sorted[None, :]
    suffix = np.zeros((n_boot, n + 1), dtype=np.float64)
    suffix[:, :-1] = np.cumsum(contrib[:, ::-1], axis=1)[:, ::-1]

    unique_tau = np.unique(scores)
    starts = np.searchsorted(s_sorted, unique_tau, side="right")
    boot_means = suffix[:, starts] / n
    mu_hat = (wt_sorted[None, :] * (s_sorted[None, :] > unique_tau[:, None])).mean(
        axis=1
    )
    apub = apub_from_bootstrap_means(boot_means, delta)
    return unique_tau, mu_hat, apub


def apub_threshold(
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    delta: float,
    n_boot: int = APUB_N_BOOT,
    seed: int = 0,
) -> Tuple[float, dict]:
    """Smallest tau with APUB_{delta}(tau) <= alpha (WACR risk budget).

    APUB follows Liu, Gao & Hu (ICML 2026) Definition 2.1 / Proposition 2.2:
    multinomial bootstrap of the weighted miss means, then the average of
    Efron ``(1-tau)``-quantiles for ``tau in (0, delta]``. ``delta`` is the
    paper's APUB level (nominal ``1-delta``); ``alpha`` is the WACR miss-risk
    constraint, as with Bernstein. Asymptotic, not PAC; tau-rule ablation only.

    Candidate thresholds are the unique calibration scores; ``APUB(tau)`` is
    non-increasing in ``tau``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    unique_tau, mu_hat, apub = apub_bounds_on_grid(
        scores, weights, delta=delta, n_boot=n_boot, seed=seed
    )
    grid_size = int(unique_tau.size)
    feasible_idx = np.nonzero(apub <= alpha)[0]
    if feasible_idx.size:
        i = int(feasible_idx[0])
        tau = float(unique_tau[i])
    else:
        i = grid_size - 1
        tau = float("inf")
    u = float(apub[i])
    mu = float(mu_hat[i])
    return tau, {
        "tau": tau,
        "mu_hat": mu,
        "U": u,
        "penalty": u - mu,
        "grid_size": grid_size,
        "n_boot": n_boot,
        "feasible": bool(feasible_idx.size),
    }


def _fit_threshold(
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    risk_bound: str,
    delta: float,
    w_test: float | None = None,
) -> Tuple[float, dict | None]:
    if risk_bound == "conformal":
        return (
            weighted_conformal_threshold(scores, weights, alpha=alpha, w_test=w_test),
            None,
        )
    if risk_bound == "apub":
        return apub_threshold(scores, weights, alpha=alpha, delta=delta)
    tau, diag = concentration_threshold(
        scores, weights, alpha=alpha, delta=delta, kind=risk_bound
    )
    return tau, diag


# ---------------------------------------------------------------------------
# Test-time retrieval evaluation
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    name: str
    tau: float
    set_sizes: np.ndarray
    contains_true: np.ndarray
    avg_set_size: float
    marginal_coverage: float
    n_eval: int = 0
    n_covered: int = 0
    r_hat: float = float("nan")  # R^ = 1 - coverage


def _coverage_from_mask(accepted: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    contains = np.zeros(accepted.shape[0], dtype=bool)
    valid = true_idx >= 0
    rows = np.where(valid)[0]
    if rows.size:
        contains[rows] = accepted[rows, true_idx[rows]]
    return contains


def true_match_rank_from_similarities(similarities: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """1-based rank of the true gallery match in descending similarity order.

    Returns -1 for queries without a valid true index.
    """
    n, m = similarities.shape
    order = np.argsort(-similarities, axis=1, kind="mergesort")
    ranks_zero_based = np.empty((n, m), dtype=np.int64)
    rows = np.arange(n)[:, None]
    ranks_zero_based[rows, order] = np.arange(m, dtype=np.int64)[None, :]

    out = np.full(n, -1, dtype=np.int64)
    valid = true_idx >= 0
    vrows = np.where(valid)[0]
    if vrows.size:
        out[vrows] = ranks_zero_based[vrows, true_idx[vrows]] + 1
    return out


def _method_result_from_evaluation(
    name: str,
    tau: float,
    set_sizes: np.ndarray,
    contains: np.ndarray,
    true_idx_test: np.ndarray,
) -> MethodResult:
    valid = true_idx_test >= 0
    n_eval = int(valid.sum())
    n_covered = int(contains[valid].sum()) if n_eval else 0
    coverage = (n_covered / n_eval) if n_eval else float("nan")
    r_hat = (1.0 - coverage) if n_eval else float("nan")
    return MethodResult(
        name=name,
        tau=tau,
        set_sizes=set_sizes,
        contains_true=contains,
        avg_set_size=float(set_sizes.mean()),
        marginal_coverage=float(coverage),
        n_eval=n_eval,
        n_covered=n_covered,
        r_hat=float(r_hat),
    )


def apply_min_set_size(
    accepted: np.ndarray, test_sims: np.ndarray, min_set_size: int
) -> np.ndarray:
    """If a query's conformal set is smaller than ``min_set_size``, add top-k by cosine."""
    if min_set_size <= 0:
        return accepted
    n_gal = accepted.shape[1]
    k = min(int(min_set_size), n_gal)
    if k <= 0:
        return accepted
    out = accepted.copy()
    short = out.sum(axis=1) < k
    if not np.any(short):
        return out
    rows = np.where(short)[0]
    top = np.argpartition(-test_sims[rows], kth=k - 1, axis=1)[:, :k]
    out[rows[:, None], top] = True
    return out


def evaluate_baseline(
    test_sims: np.ndarray,
    tau: float,
    true_idx_test: np.ndarray,
    *,
    min_set_size: int = 0,
) -> MethodResult:
    """Method A: accept gallery item iff ``(1 - sim) <= tau``."""
    accepted = apply_min_set_size((1.0 - test_sims) <= tau, test_sims, min_set_size)
    set_sizes = accepted.sum(axis=1).astype(np.int64)
    contains = _coverage_from_mask(accepted, true_idx_test)
    return _method_result_from_evaluation(
        "Baseline (Method A)", tau, set_sizes, contains, true_idx_test
    )


def evaluate_adaptive(
    test_sims: np.ndarray,
    tau: float,
    phi_test: np.ndarray,
    true_idx_test: np.ndarray,
    *,
    min_set_size: int = 0,
) -> MethodResult:
    """Method B: accept gallery item iff ``(1 - sim) <= tau * Phi(x_test)``."""
    per_query_tau = (tau * phi_test).reshape(-1, 1)
    accepted = apply_min_set_size(
        (1.0 - test_sims) <= per_query_tau, test_sims, min_set_size
    )
    set_sizes = accepted.sum(axis=1).astype(np.int64)
    contains = _coverage_from_mask(accepted, true_idx_test)
    return _method_result_from_evaluation(
        "Risk-Aware (Method B)", tau, set_sizes, contains, true_idx_test
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _print_result(r: MethodResult, *, alpha: float) -> None:
    print(f"[{r.name}]")
    print(f"  tau            = {r.tau:.6f}")
    print(f"  Average Retrieval Set Size = {r.avg_set_size:.3f}")
    print(f"  marginal coverage          = {r.marginal_coverage:.4f}")
    print(
        f"  set size stats : min={int(r.set_sizes.min())}  "
        f"median={int(np.median(r.set_sizes))}  "
        f"max={int(r.set_sizes.max())}"
    )
    if r.n_eval > 0:
        print(f"  covered       = {r.n_covered}/{r.n_eval} = {r.marginal_coverage:.4f}")
        print(f"  R^            = {r.r_hat:.4f}  (R^ = 1 - coverage)")
        status = "RISK_CONTROLLED" if r.r_hat <= alpha else "RISK_NOT_CONTROLLED"
        print(f"  risk control  : R^ <= alpha={alpha}  ->  {status}")
    else:
        print("  risk control  : UNKNOWN (no valid test queries with gallery match)")
    print()


def save_side_by_side_csv(
    output_csv: Path,
    *,
    query_paths: List[str],
    true_idx: np.ndarray,
    phi_test: np.ndarray,
    max_sim_test: np.ndarray,
    true_rank: np.ndarray,
    baseline: MethodResult,
    adaptive: MethodResult,
) -> None:
    """Write per-query baseline vs adaptive retrieval metrics side-by-side."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query_path",
                "true_gallery_index",
                "true_rank_baseline",
                "true_rank_adaptive",
                "phi_test",
                "max_similarity_test",
                "baseline_set_size",
                "baseline_contains_true",
                "adaptive_set_size",
                "adaptive_contains_true",
                "tau_base",
                "tau_adaptive",
            ]
        )
        for i, qpath in enumerate(query_paths):
            writer.writerow(
                [
                    qpath,
                    int(true_idx[i]),
                    int(true_rank[i]),
                    int(true_rank[i]),
                    float(phi_test[i]),
                    float(max_sim_test[i]),
                    int(baseline.set_sizes[i]),
                    int(baseline.contains_true[i]),
                    int(adaptive.set_sizes[i]),
                    int(adaptive.contains_true[i]),
                    float(baseline.tau),
                    float(adaptive.tau),
                ]
            )
    print(f"[Output] wrote side-by-side CSV: {output_csv}")


def _fit_phi(
    *,
    uncertainty_method: str,
    beta: float,
    calib: EmbeddingBundle,
    test: EmbeddingBundle,
    calib_sims: np.ndarray,
    test_sims: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``UncertaintyAdapter`` and return ``(phi_cal, phi_test)``."""
    if uncertainty_method == "mcd":
        if calib.fu is None or test.fu is None:
            raise ValueError(
                "uncertainty_method='mcd' requires 'fu' in both calibration and test query .npz files"
            )
        adapter, _ = UncertaintyAdapter.fit(
            method="mcd", beta=beta, cal_fu=calib.fu
        )
        return adapter.transform(calib.fu), adapter.transform(test.fu)

    adapter, cal_fu_used = UncertaintyAdapter.fit(
        method="top1", beta=beta, cal_sims=calib_sims
    )
    test_fu = top1_fu_from_similarities(test_sims)
    return adapter.transform(cal_fu_used), adapter.transform(test_fu)


def run_pipeline(
    *,
    calib_queries_npz: Path,
    test_queries_npz: Path,
    calib_gallery_npz: Path,
    test_gallery_npz: Path,
    importance_weights_path: Path,
    alpha: float,
    beta: float,
    uncertainty_method: str,
    save_csv: bool,
    csv_path: Path,
    alpha_cal: float | None = None,
    min_set_size: int = 0,
    risk_bound: str = "conformal",
    delta: float = 0.1,
) -> Dict[str, MethodResult]:
    calib = load_bundle(calib_queries_npz, name="TinyFace_Queries")
    test = load_bundle(test_queries_npz, name="SCFace_Queries")
    calib_gallery = load_bundle(calib_gallery_npz, name="TinyFace_Gallery")
    test_gallery = load_bundle(test_gallery_npz, name="SCFace_Gallery")
    weights = load_importance_weights(importance_weights_path, calib.paths)
    w_test = importance_weight_cap(importance_weights_path, weights)

    w_sum = float(weights.sum())
    ess = effective_sample_size(weights)
    print(f"[Weights] file={importance_weights_path.name}")
    print(
        f"  w: min={weights.min():.4f}  mean={weights.mean():.4f}  "
        f"max={weights.max():.4f}  sum={w_sum:.4f}  ESS={ess:.2f}  "
        f"test atom B={w_test:.2f}"
    )
    print(
        f"[Data] calib queries={len(calib.paths)}  test queries={len(test.paths)}  "
        f"calib gallery={len(calib_gallery.paths)}  test gallery={len(test_gallery.paths)}"
    )

    true_calib = true_match_indices(calib, calib_gallery)
    true_test = true_match_indices(test, test_gallery)
    print(
        f"[GT] calib matches: {(true_calib >= 0).sum()}/{len(calib.paths)}  "
        f"test matches: {(true_test >= 0).sum()}/{len(test.paths)}"
    )

    valid = true_calib >= 0
    if not np.all(valid):
        print(f"[GT] dropping {int((~valid).sum())} calibration queries without a match")
        calib = replace(
            calib,
            embeddings=calib.embeddings[valid],
            paths=[p for p, v in zip(calib.paths, valid) if v],
            fu=(calib.fu[valid] if calib.fu is not None else None),
        )
        weights = weights[valid]
        true_calib = true_calib[valid]

    calib_sims = cosine_similarity_matrix(calib.embeddings, calib_gallery.embeddings)
    test_sims = cosine_similarity_matrix(test.embeddings, test_gallery.embeddings)
    max_sim_test = np.max(test_sims, axis=1)
    true_rank_test = true_match_rank_from_similarities(test_sims, true_test)

    s_base_calib = (1.0 - calib_sims[np.arange(calib_sims.shape[0]), true_calib]).astype(
        np.float64
    )

    phi_cal, phi_test = _fit_phi(
        uncertainty_method=uncertainty_method,
        beta=beta,
        calib=calib,
        test=test,
        calib_sims=calib_sims,
        test_sims=test_sims,
    )
    print(f"[Uncertainty] method={uncertainty_method}")
    print(
        f"[Uncertainty] Phi(cal): mean={phi_cal.mean():.4f} "
        f"min={phi_cal.min():.4f} max={phi_cal.max():.4f}  |  "
        f"Phi(test): mean={phi_test.mean():.4f} "
        f"min={phi_test.min():.4f} max={phi_test.max():.4f}"
    )

    s_adaptive_calib = s_base_calib / phi_cal
    alpha_for_tau = float(alpha if alpha_cal is None else alpha_cal)
    if not 0.0 < alpha_for_tau < 1.0:
        raise ValueError(f"alpha_cal must be in (0, 1); got {alpha_for_tau}")
    if risk_bound not in {"conformal", "bernstein", "hoeffding", "apub"}:
        raise ValueError(
            f"risk_bound must be conformal, bernstein, hoeffding, or apub; got {risk_bound!r}"
        )
    tau_base, diag_base = _fit_threshold(
        s_base_calib,
        weights,
        alpha=alpha_for_tau,
        risk_bound=risk_bound,
        delta=delta,
        w_test=w_test,
    )
    tau_adaptive, diag_ad = _fit_threshold(
        s_adaptive_calib,
        weights,
        alpha=alpha_for_tau,
        risk_bound=risk_bound,
        delta=delta,
        w_test=w_test,
    )
    print(
        f"[Calibration] bound={risk_bound}  alpha_cal={alpha_for_tau}  "
        f"delta={delta}  min_set_size={min_set_size}"
    )
    print(f"  tau_base={tau_base:.6f}  tau_adaptive={tau_adaptive:.6f}")
    for name, diag in (("Method A", diag_base), ("Method B", diag_ad)):
        if diag is None:
            continue
        feas = "feasible" if diag["feasible"] else "INFEASIBLE (tau=+inf, whole gallery)"
        print(
            f"  {name}: U={diag['U']:.4f}  mu={diag['mu_hat']:.4f}  "
            f"penalty={diag['penalty']:.4f}  {feas}"
        )
    print()

    results = {
        "baseline": evaluate_baseline(
            test_sims, tau_base, true_test, min_set_size=min_set_size
        ),
        "adaptive": evaluate_adaptive(
            test_sims, tau_adaptive, phi_test, true_test, min_set_size=min_set_size
        ),
    }
    print(f"[Risk target] alpha={alpha}  (tau fit at alpha_cal={alpha_for_tau})")
    print()
    for r in results.values():
        _print_result(r, alpha=alpha)

    if save_csv:
        save_side_by_side_csv(
            csv_path,
            query_paths=test.paths,
            true_idx=true_test,
            true_rank=true_rank_test,
            phi_test=phi_test,
            max_sim_test=max_sim_test,
            baseline=results["baseline"],
            adaptive=results["adaptive"],
        )
    return results


def resolve_data_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve NPZ paths from ``--pair`` or explicit CLI paths."""
    if args.pair:
        combination = args.pair.strip()
        cache_dir = (CACHE_DIR / combination).resolve()
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Pair cache directory not found: {cache_dir}")
        paths = {
            "calib_queries": cache_dir / f"{args.cal}_Queries_mcd.npz",
            "test_queries": cache_dir / f"{args.test}_Queries_mcd.npz",
            "calib_gallery": cache_dir / f"{args.cal}_Gallery_mcd.npz",
            "test_gallery": cache_dir / f"{args.test}_Gallery_mcd.npz",
            "weights": cache_dir / args.weights_file,
        }
    else:
        paths = {
            "calib_queries": resolve_path(args.calib_queries),
            "test_queries": resolve_path(args.test_queries),
            "calib_gallery": resolve_path(args.calib_gallery),
            "test_gallery": resolve_path(args.test_gallery),
            "weights": resolve_path(args.weights_file),
        }

    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required cache file(s):\n  " + "\n  ".join(missing)
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help=(
            "Folder under cache/ (e.g. 'tinyface_scface'). "
            "When set, resolves query/gallery/weights paths under repo cache/."
        ),
    )
    parser.add_argument(
        "--cal",
        default="TinyFace",
        help="Calibration dataset prefix when using --pair (default: TinyFace).",
    )
    parser.add_argument(
        "--test",
        default="SCFace",
        help="Test dataset prefix when using --pair (default: SCFace).",
    )
    parser.add_argument(
        "--calib-queries",
        default="cache/tinyface_scface/TinyFace_Queries_mcd.npz",
        help="Calibration queries .npz (ignored if --pair is set).",
    )
    parser.add_argument(
        "--test-queries",
        default="cache/tinyface_scface/SCFace_Queries_mcd.npz",
        help="Test queries .npz (ignored if --pair is set).",
    )
    parser.add_argument(
        "--calib-gallery",
        default="cache/tinyface_scface/TinyFace_Gallery_mcd.npz",
        help="Calibration gallery .npz (ignored if --pair is set).",
    )
    parser.add_argument(
        "--test-gallery",
        default="cache/tinyface_scface/SCFace_Gallery_mcd.npz",
        help="Test gallery .npz (ignored if --pair is set).",
    )
    parser.add_argument(
        "--weights-file",
        "--importance-weights",
        "--kmm-file",
        "--kmm-weights",
        dest="weights_file",
        default=DEFAULT_WEIGHTS_FILE,
        help=(
            "Calibration importance weights (.npz with keys w, paths), filename "
            f"under cache/<combination>/ or a full path (wx, KMM, KL, …). "
            f"Default: {DEFAULT_WEIGHTS_FILE}."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--risk-bound",
        choices=["conformal", "bernstein", "hoeffding", "apub"],
        default="conformal",
        help=(
            "How tau is chosen. 'conformal' (default) is the weighted quantile. "
            "'bernstein' / 'hoeffding' pick the smallest tau with a Bonferroni "
            "upper bound U(tau) <= alpha on weighted miss risk. 'apub' uses the "
            "Liu-Gao-Hu average-percentile UCB (Def. 2.1; ablation; asymptotic)."
        ),
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help=(
            "Bernstein/Hoeffding failure probability. For APUB this is the paper "
            "level alpha (nominal coverage 1-delta); ignored by conformal."
        ),
    )
    parser.add_argument(
        "--alpha-cal",
        type=float,
        default=None,
        help=(
            "Alpha used only to fit tau (default: same as --alpha). "
            "Set lower than --alpha for a more conservative threshold. "
            "Ignored if --alpha-cal-ratio is set."
        ),
    )
    parser.add_argument(
        "--alpha-cal-ratio",
        type=float,
        default=None,
        help="If set, tau is fit at (--alpha * ratio), e.g. 0.5 -> alpha/2.",
    )
    parser.add_argument(
        "--min-set-size",
        type=int,
        default=0,
        help=(
            "If a query's conformal set is smaller than this, add the top-k "
            "gallery images by cosine similarity (k=1 avoids empty sets)."
        ),
    )
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--uncertainty-method",
        choices=["top1", "mcd"],
        default="top1",
        help="Uncertainty source for Phi(x) in Method B: top1-distance or MCD fu.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save per-query baseline vs adaptive metrics side-by-side.",
    )
    parser.add_argument(
        "--csv-path",
        default="WACR/per_query_results.csv",
        help="Output CSV path used when --save-csv is provided (relative to repo root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_data_paths(args)

    if args.pair:
        print(f"Pair        : {args.pair.strip()}")
        print(f"Calibration : {args.cal}")
        print(f"Test        : {args.test}")
    print(f"Weights file: {paths['weights'].name}")

    alpha_cal = args.alpha_cal
    if args.alpha_cal_ratio is not None:
        alpha_cal = float(args.alpha) * float(args.alpha_cal_ratio)

    run_pipeline(
        calib_queries_npz=paths["calib_queries"],
        test_queries_npz=paths["test_queries"],
        calib_gallery_npz=paths["calib_gallery"],
        test_gallery_npz=paths["test_gallery"],
        importance_weights_path=paths["weights"],
        alpha=args.alpha,
        alpha_cal=alpha_cal,
        min_set_size=args.min_set_size,
        beta=args.beta,
        uncertainty_method=args.uncertainty_method,
        save_csv=args.save_csv,
        csv_path=resolve_path(args.csv_path),
        risk_bound=args.risk_bound,
        delta=args.delta,
    )


if __name__ == "__main__":
    main()
