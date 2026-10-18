#!/usr/bin/env python3
"""
KLIEP importance weights for covariate-shift conformal prediction.

Problem
-------
Under covariate shift, calibration points x_i ~ p_source while test queries come
from p_test. Reweighted CP needs w_i ∝ p_test(x_i) / p_source(x_i).

KLIEP (Kullback-Leibler Importance Estimation Procedure; Sugiyama et al.) fits a
parametric density-ratio model by maximizing test log-likelihood subject to a
source-side normalization constraint.

Mathematics
-----------
Embeddings are L2-normalized. Sample b basis centers {c_l} from test points.
Gaussian (RBF) basis functions:

    phi_l(x) = exp(-||x - c_l||^2 / (2 sigma^2))

Model the density ratio as a non-negative linear combination:

    w(x; alpha) = sum_l alpha_l * phi_l(x),   alpha_l >= 0

**Normalization constraint** (average weight over source equals 1):

    (1/n) sum_i w(x_i; alpha) = 1
    <=>  b^T alpha = 1,   where b_l = (1/n) sum_i phi_l(x_i^calib)

**Objective** (maximize mean test log-likelihood):

    max_alpha  (1/m) sum_j log( sum_l A_{jl} alpha_l )
    s.t.       b^T alpha = 1,  alpha >= 0

where A_{jl} = phi_l(x_j^test). Solved via convex optimization (cvxpy) with a
projected-gradient fallback.

**Per-calibration weights**::

    w_i = w(x_i^calib; alpha) = Phi_calib @ alpha

**Post-processing** (``finalize_wcp_weights``)::

    w_i <- min(w_i, w_max)
    iteratively rescale uncapped entries so sum(w) ≈ n_calib

Bandwidth sigma is chosen by ``--sigma median`` (median pairwise distance),
``--sigma cv`` (held-out test log-likelihood over a sigma grid), or a fixed float.

Full flow
---------
1. Load ``<cal>_Queries_mcd.npz`` (source) and ``<test>_Queries_mcd.npz`` (target).
2. L2-normalize embeddings; pick sigma (median, CV, or fixed).
3. Sample ``--n-basis`` Gaussian centers from test embeddings.
4. Build design matrices Phi_calib, Phi_test; compute constraint vector b.
5. Fit alpha by maximizing test log-likelihood subject to b^T alpha = 1, alpha >= 0.
6. Evaluate w_i = Phi_calib @ alpha at each calibration point.
7. Clip at ``w_max``; rescale so sum(w) ≈ n_calib.
8. Write ``<cal>_Queries_kliep.npz`` with ``w``, ``paths``, ``alpha``, and metadata.

Weight convention
-----------------
sum(w) ≈ n_calib, matching KL/KMM/RuLSIF caches.

Example (from repo root)::

    python Inference/compute_kliep_weights.py --combination tinyface_scface
    python Inference/compute_kliep_weights.py --combination tinyface_scface --sigma cv --n-basis 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvxpy as cp
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wacr.embeddings import load_embeddings, load_embeddings_and_paths
from wacr.paths import (
    DEFAULT_CALIB_DATASET,
    DEFAULT_TEST_DATASET,
    add_combination_argument,
    normalize_combination_name,
    resolve_weight_script_paths,
)
from wacr.similarity import l2_normalize
from wacr.weights import finalize_wcp_weights


def median_bandwidth(
    x: np.ndarray,
    *,
    max_samples: int = 800,
    seed: int = 0,
) -> float:
    """Median pairwise distance heuristic on a subsample (RBF bandwidth)."""
    n = x.shape[0]
    if n < 2:
        return 1.0
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        xs = x[idx]
    else:
        xs = x
    d = pairwise_distances(xs, xs, metric="euclidean")
    triu = d[np.triu_indices(xs.shape[0], k=1)]
    if triu.size == 0:
        return 1.0
    med = float(np.median(triu))
    return med if med > 1e-8 else 1.0


def _rbf_design(points: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian basis design matrix Phi[i, l] = exp(-||p_i - c_l||^2 / (2 sigma^2))."""
    if sigma <= 0.0:
        raise ValueError(f"bandwidth sigma must be > 0; got {sigma}")
    gamma = 1.0 / (2.0 * sigma * sigma)
    return rbf_kernel(points, centers, gamma=gamma)


def _select_centers(
    x_test: np.ndarray, n_basis: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample up to ``n_basis`` distinct test points as Gaussian basis centers."""
    n_test = x_test.shape[0]
    b = int(min(n_basis, n_test))
    if b < 1:
        raise ValueError("test set is empty; cannot place KLIEP basis centers")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_test, size=b, replace=False)
    idx.sort()
    return x_test[idx], idx


def _solve_alpha_cvxpy(A: np.ndarray, b_vec: np.ndarray) -> np.ndarray | None:
    """Solve max sum(log(A alpha)) s.t. b_vec @ alpha == 1, alpha >= 0 via cvxpy."""
    n_test, b = A.shape
    alpha = cp.Variable(b)
    objective = cp.Maximize(cp.sum(cp.log(A @ alpha)) / n_test)
    constraints = [b_vec @ alpha == 1.0, alpha >= 0.0]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
    except cp.error.SolverError:
        return None
    if alpha.value is None:
        return None
    return np.asarray(alpha.value, dtype=np.float64).reshape(-1)


def _solve_alpha_projected_gradient(
    A: np.ndarray,
    b_vec: np.ndarray,
    *,
    max_iter: int = 5000,
    lr: float = 0.1,
    tol: float = 1e-7,
) -> np.ndarray:
    """Projected-gradient ascent fallback when cvxpy is unavailable or fails."""
    n_test, b = A.shape
    b_norm_sq = float(b_vec @ b_vec)
    if b_norm_sq <= 0.0:
        raise RuntimeError("KLIEP basis collapsed (b_vec is zero); adjust bandwidth.")

    alpha = b_vec / b_norm_sq
    alpha = np.maximum(alpha, 1e-12)

    prev_obj = -np.inf
    for _ in range(max_iter):
        Aa = np.maximum(A @ alpha, 1e-12)
        grad = A.T @ (1.0 / Aa)
        alpha = alpha + (lr / n_test) * grad
        alpha = alpha + b_vec * ((1.0 - b_vec @ alpha) / b_norm_sq)
        alpha = np.maximum(alpha, 0.0)
        denom = float(b_vec @ alpha)
        if denom > 1e-12:
            alpha = alpha / denom
        Aa = np.maximum(A @ alpha, 1e-12)
        obj = float(np.sum(np.log(Aa)))
        if np.abs(obj - prev_obj) <= tol * (1.0 + np.abs(prev_obj)):
            break
        prev_obj = obj
    return alpha


def _fit_alpha(A: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, str]:
    """Fit KLIEP coefficients, preferring cvxpy and falling back to projected gradient."""
    alpha = _solve_alpha_cvxpy(A, b_vec)
    if alpha is not None and np.all(np.isfinite(alpha)):
        alpha = np.maximum(alpha, 0.0)
        if alpha.sum() > 0.0:
            return alpha, "cvxpy"
    alpha = _solve_alpha_projected_gradient(A, b_vec)
    return alpha, "projected_gradient"


def _test_log_likelihood(A: np.ndarray, alpha: np.ndarray) -> float:
    """Mean test log-likelihood (1/n) sum_j log(A[j] @ alpha)."""
    Aa = np.maximum(A @ alpha, 1e-300)
    return float(np.mean(np.log(Aa)))


def _sigma_grid(base_sigma: float) -> list[float]:
    """Candidate bandwidths around the median heuristic for cross-validation."""
    factors = (0.2, 0.5, 1.0, 2.0, 5.0)
    return [base_sigma * f for f in factors if base_sigma * f > 0.0]


def select_sigma_cv(
    x_calib: np.ndarray,
    x_test: np.ndarray,
    *,
    n_basis: int,
    seed: int,
    n_folds: int = 5,
) -> float:
    """Pick bandwidth by held-out test log-likelihood over a sigma grid."""
    n_test = x_test.shape[0]
    base = median_bandwidth(np.vstack([x_calib, x_test]), seed=seed)
    candidates = _sigma_grid(base)
    if n_test < n_folds or n_test < 2:
        return base

    centers, _ = _select_centers(x_test, n_basis, seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_test)
    folds = np.array_split(perm, n_folds)

    best_sigma = base
    best_score = -np.inf
    for sigma in candidates:
        b_vec = _rbf_design(x_calib, centers, sigma).mean(axis=0)
        fold_scores = []
        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            if train_idx.size < 1 or val_idx.size < 1:
                continue
            A_train = _rbf_design(x_test[train_idx], centers, sigma)
            try:
                alpha, _ = _fit_alpha(A_train, b_vec)
            except RuntimeError:
                continue
            A_val = _rbf_design(x_test[val_idx], centers, sigma)
            fold_scores.append(_test_log_likelihood(A_val, alpha))
        if not fold_scores:
            continue
        score = float(np.mean(fold_scores))
        if score > best_score:
            best_score = score
            best_sigma = sigma
    return best_sigma


def kliep_weights(
    x_calib: np.ndarray,
    x_test: np.ndarray,
    *,
    n_basis: int = 100,
    sigma: float | str = "median",
    w_max: float = 30.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, int, str]:
    """Estimate w_i = p_test(x_i)/p_source(x_i) at calibration points via KLIEP."""
    x_calib = l2_normalize(np.asarray(x_calib, dtype=np.float64))
    x_test = l2_normalize(np.asarray(x_test, dtype=np.float64))
    n_calib = x_calib.shape[0]

    if isinstance(sigma, str):
        if sigma.lower() == "median":
            sigma_used = median_bandwidth(np.vstack([x_calib, x_test]), seed=seed)
        elif sigma.lower() == "cv":
            sigma_used = select_sigma_cv(x_calib, x_test, n_basis=n_basis, seed=seed)
        else:
            sigma_used = float(sigma)
    else:
        sigma_used = float(sigma)

    centers, _ = _select_centers(x_test, n_basis, seed)
    n_basis_used = centers.shape[0]

    A = _rbf_design(x_test, centers, sigma_used)
    b_vec = _rbf_design(x_calib, centers, sigma_used).mean(axis=0)

    alpha, solver = _fit_alpha(A, b_vec)

    Phi_calib = _rbf_design(x_calib, centers, sigma_used)
    w_raw = Phi_calib @ alpha
    w = finalize_wcp_weights(w_raw, n_calib, w_max=w_max)
    return w, alpha, sigma_used, n_basis_used, solver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "KLIEP density-ratio importance weights for calibration queries "
            "relative to a test embedding distribution."
        )
    )
    add_combination_argument(parser)
    parser.add_argument(
        "--cal-dataset",
        type=str,
        default=DEFAULT_CALIB_DATASET,
        help=f"Source/calibration dataset name (default: {DEFAULT_CALIB_DATASET}).",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default=DEFAULT_TEST_DATASET,
        help=f"Target/test dataset name (default: {DEFAULT_TEST_DATASET}).",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Override calibration .npz path.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Override test .npz path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output .npz path.",
    )
    parser.add_argument(
        "--n-basis",
        type=int,
        default=100,
        help="Number of Gaussian basis functions (centers sampled from test).",
    )
    parser.add_argument(
        "--sigma",
        type=str,
        default="median",
        help="Gaussian bandwidth: float, 'median' (default), or 'cv'.",
    )
    parser.add_argument(
        "--w-max",
        type=float,
        default=30.0,
        help="Upper cap on each weight before renormalization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for basis-center sampling and CV folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combination = normalize_combination_name(args.combination)

    calib_path, test_path, output_path = resolve_weight_script_paths(
        combination,
        weight_kind="kliep",
        cal_dataset=args.cal_dataset,
        test_dataset=args.test_dataset,
        calib=args.calib,
        test_npz=args.test,
        output=args.output,
    )

    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    sigma_arg: float | str
    sl = str(args.sigma).lower()
    if sl in ("median", "cv"):
        sigma_arg = sl
    else:
        sigma_arg = float(args.sigma)

    print(f"Pair: {combination}")
    print(f"Loading calibration embeddings: {calib_path}")
    x_calib, calib_paths = load_embeddings_and_paths(calib_path)
    print(f"  shape: {x_calib.shape}, paths: {len(calib_paths)}")

    print(f"Loading test embeddings: {test_path}")
    x_test = load_embeddings(test_path)
    print(f"  shape: {x_test.shape}")

    if x_calib.shape[1] != x_test.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: calib D={x_calib.shape[1]}, "
            f"test D={x_test.shape[1]}"
        )

    print(
        f"Running KLIEP (n_basis={args.n_basis}, sigma={args.sigma}, "
        f"w_max={args.w_max}) ..."
    )
    weights, alpha, sigma_used, n_basis_used, solver = kliep_weights(
        x_calib,
        x_test,
        n_basis=int(args.n_basis),
        sigma=sigma_arg,
        w_max=float(args.w_max),
        seed=int(args.seed),
    )
    weights = np.asarray(weights, dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        w=weights,
        paths=calib_paths,
        calib_npz=str(calib_path),
        test_npz=str(test_path),
        method=np.array("kliep"),
        solver=np.array(solver),
        sigma=np.float32(sigma_used),
        n_basis=np.int32(n_basis_used),
        alpha=np.asarray(alpha, dtype=np.float32),
        w_max=np.float32(args.w_max),
        seed=np.int32(args.seed),
    )

    print(f"Saved KLIEP weights to: {output_path}")
    print(f"  solver: {solver}  sigma: {sigma_used:.6f}  n_basis: {n_basis_used}")
    print(
        f"  w: min={weights.min():.6f}, max={weights.max():.6f}, "
        f"mean={weights.mean():.6f}, sum={weights.sum():.6f}"
    )


if __name__ == "__main__":
    main()
