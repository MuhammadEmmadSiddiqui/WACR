#!/usr/bin/env python3
"""
Kernel Mean Matching (KMM) importance weights for covariate-shift conformal prediction.

Problem
-------
Under covariate shift, calibration points x_i ~ p_source while test queries come
from p_test. Reweighted CP needs w_i ∝ p_test(x_i) / p_source(x_i).

KMM (Gretton et al.) finds non-negative weights on calibration points so that
the weighted source mean in a reproducing-kernel Hilbert space (RKHS) matches
the unweighted test mean — a moment-matching view of the density ratio.

Reference: "KMM-CP: Practical Conformal Prediction under Covariate Shift via
Selective Kernel Mean Matching" (arXiv:2603.26415).

Mathematics
-----------
Embeddings are L2-normalized. RBF kernel k(x, x') = exp(-gamma ||x - x'||^2).
``gamma=None`` keeps sklearn's 1/n_features default; ``--gamma median`` uses
1 / median pairwise squared Euclidean distance. Let K_ss[i,j] = k(x_i^calib,
x_j^calib) and K_st[i,j] = k(x_i^calib, x_j^test).

**Standard KMM** (``--method standard``) solves the convex QP::

    min_w   (1/2) w^T (K_ss + eps I) w - kappa^T w
    s.t.    0 <= w_i <= B,
            n(1 - epsilon) <= sum_i w_i <= n(1 + epsilon)

where n = n_calib, m = n_test, and::

    kappa_i = (n/m) * sum_j K_st[i,j]

The objective matches weighted source embeddings to the scaled test kernel mean;
constraints keep weights non-negative, bounded by ``--B``, and approximately
normalized (sum(w) ≈ n within ``--epsilon``).

**Selective KMM** (``--method selective``) extends this in two stages:

Stage 1 — Joint QP over calibration weights w and test selection variables alpha::

    min_{w, alpha}  (1/2) z^T P z
    s.t.            same w constraints as above,
                    0 <= alpha_j <= 1,
                    m * tau <= sum_j alpha_j <= m

where z = [w; alpha] and P is built from blocks K_ss, K_st, K_tt with scale n/m.
This jointly optimizes weights and soft test-point selection.

Stage 2 — Hard-select the top ceil(m * tau) test points by alpha, then re-run
standard KMM on the retained subset.

Full flow
---------
1. Load ``<cal>_Queries_mcd.npz`` (source) and ``<test>_Queries_mcd.npz`` (target).
2. L2-normalize embedding rows.
3. Build RBF Gram matrices; solve the KMM QP (standard or selective two-stage).
4. Clip any solver negatives to 0 and renormalize via ``finalize_wcp_weights``.
5. Write ``<cal>_Queries_kmm_{standard|selective}.npz`` with ``w``, ``paths``,
   and metadata (``B``, ``epsilon``, ``tau``).

Weight convention
-----------------
sum(w) ≈ n_calib (enforced by QP constraints), matching KL/KLIEP/RuLSIF caches.

Example (from repo root)::

    python Inference/compute_kmm_weights.py --combination tinyface_scface
    python Inference/compute_kmm_weights.py --combination tinyface_scface --method selective --B 30 --tau 0.5
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cvxpy as cp
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel

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

# Dense KMM builds an N×N kernel; warn above this calibration count.
KMM_SIZE_WARN_THRESHOLD = 3000


def _warn_if_large(n_points: int, what: str) -> None:
    if n_points > KMM_SIZE_WARN_THRESHOLD:
        warnings.warn(
            f"KMM on {n_points} {what} points: the dense {n_points}×{n_points} "
            f"kernel/QP (~{8 * n_points ** 2 / 1e6:.0f} MB) may be slow or hit "
            f"OOM. Consider subsampling to <= {KMM_SIZE_WARN_THRESHOLD} points.",
            stacklevel=2,
        )


def median_rbf_gamma(
    X: np.ndarray, *, n_sample: int = 800, seed: int = 0
) -> float:
    """gamma = 1 / median pairwise squared Euclidean distance (median heuristic)."""
    X = l2_normalize(np.asarray(X, dtype=np.float64))
    n = int(X.shape[0])
    m = min(int(n_sample), n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    d2 = euclidean_distances(X[idx], X[idx], squared=True)
    tri = d2[np.triu_indices(m, k=1)]
    med = float(np.median(tri))
    return 1.0 / max(med, 1e-12)


def _rbf_kw(gamma: float | None) -> dict:
    return {} if gamma is None else {"gamma": float(gamma)}


def get_kmm_weights(
    X_calib: np.ndarray,
    X_test: np.ndarray,
    B: float = 30.0,
    epsilon: float = 0.1,
    gamma: float | None = None,
) -> np.ndarray:
    """Standard KMM (paper Equations 5–6).

    ``gamma`` is the RBF coefficient. ``None`` keeps sklearn's 1/n_features
    default (too wide on 512-D unit-norm embeddings). Pass the median heuristic
    for a kernel that actually varies.
    """
    X_calib = l2_normalize(np.asarray(X_calib, dtype=np.float64))
    X_test = l2_normalize(np.asarray(X_test, dtype=np.float64))

    n_calib = X_calib.shape[0]
    n_test = X_test.shape[0]

    _warn_if_large(n_calib, "calibration")

    kw = _rbf_kw(gamma)
    K = rbf_kernel(X_calib, X_calib, **kw)
    K_test = rbf_kernel(X_calib, X_test, **kw)
    kappa = (n_calib / n_test) * np.sum(K_test, axis=1)

    w = cp.Variable(n_calib)
    objective = cp.Minimize(
        0.5 * cp.quad_form(w, K + 1e-6 * np.eye(n_calib)) - kappa.T @ w
    )
    constraints = [
        w >= 0,
        w <= B,
        cp.sum(w) >= n_calib * (1 - epsilon),
        cp.sum(w) <= n_calib * (1 + epsilon),
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if w.value is None:
        raise RuntimeError(f"KMM optimization failed with status: {prob.status}")

    return np.asarray(w.value, dtype=np.float64).reshape(-1)


def get_selective_kmm_weights(
    X_calib: np.ndarray,
    X_test: np.ndarray,
    B: float = 30.0,
    epsilon: float = 0.1,
    tau: float = 0.5,
    gamma: float | None = None,
) -> np.ndarray:
    """
    Selective KMM (paper Algorithm 1): joint optimization with yield constraint,
    then threshold test selection and re-run standard KMM on the retained subset.
    """
    X_calib = l2_normalize(np.asarray(X_calib, dtype=np.float64))
    X_test = l2_normalize(np.asarray(X_test, dtype=np.float64))

    n_calib = X_calib.shape[0]
    n_test = X_test.shape[0]

    _warn_if_large(n_calib + n_test, "joint (calibration + test)")

    kw = _rbf_kw(gamma)
    K_ss = rbf_kernel(X_calib, X_calib, **kw)
    K_st = rbf_kernel(X_calib, X_test, **kw)
    K_tt = rbf_kernel(X_test, X_test, **kw)

    scale = n_calib / n_test
    P_top = np.hstack([K_ss, -scale * K_st])
    P_bottom = np.hstack([-scale * K_st.T, (scale**2) * K_tt])
    P = np.vstack([P_top, P_bottom])
    P = (P + P.T) / 2.0
    P += 1e-4 * np.eye(n_calib + n_test)

    z = cp.Variable(n_calib + n_test)
    w = z[:n_calib]
    alpha = z[n_calib:]

    objective = cp.Minimize(0.5 * cp.quad_form(z, P))
    constraints = [
        w >= 0,
        w <= B,
        cp.sum(w) >= n_calib * (1 - epsilon),
        cp.sum(w) <= n_calib * (1 + epsilon),
        alpha >= 0,
        alpha <= 1,
        cp.sum(alpha) >= n_test * tau,
        cp.sum(alpha) <= n_test,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if w.value is None or alpha.value is None:
        raise RuntimeError(f"Selective KMM (Stage 1) failed with status: {prob.status}")

    alpha_vals = np.asarray(alpha.value, dtype=np.float64).reshape(-1)

    n_select = max(2, int(np.ceil(n_test * tau)))
    n_select = min(n_select, n_test)
    selected_idx = np.argsort(alpha_vals)[::-1][:n_select]
    X_test_selected = X_test[selected_idx]

    return get_kmm_weights(
        X_calib, X_test_selected, B=B, epsilon=epsilon, gamma=gamma
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute KMM weights mapping calibration embeddings to a target distribution."
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
        help=(
            "Override base output path; method suffix is appended "
            "(e.g. TinyFace_Queries_kmm_standard.npz)."
        ),
    )
    parser.add_argument("--B", type=float, default=30.0, help="Upper bound on each weight.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Tolerance on sum(w) ≈ n_calib.",
    )
    parser.add_argument(
        "--method",
        choices=["standard", "selective"],
        default="standard",
        help="KMM variant to optimize.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Target selection fraction for selective KMM.",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="default",
        help="RBF gamma: 'default' (1/n_features), 'median', or a float.",
    )
    return parser.parse_args()


def _parse_gamma(raw: str, X_calib: np.ndarray, X_test: np.ndarray) -> float | None:
    token = str(raw).strip().lower()
    if token in ("default", "sklearn", "none"):
        return None
    if token == "median":
        stacked = np.vstack(
            [
                l2_normalize(np.asarray(X_calib, dtype=np.float64)),
                l2_normalize(np.asarray(X_test, dtype=np.float64)),
            ]
        )
        return median_rbf_gamma(stacked)
    return float(raw)


def main() -> None:
    args = parse_args()
    combination = normalize_combination_name(args.combination)

    calib_path, test_path, output_path = resolve_weight_script_paths(
        combination,
        weight_kind="kmm",
        cal_dataset=args.cal_dataset,
        test_dataset=args.test_dataset,
        calib=args.calib,
        test_npz=args.test,
        output=args.output,
    )
    extra = ""
    gamma_token = str(args.gamma).strip().lower()
    if gamma_token == "median":
        extra = "_median"
    output_path = output_path.with_name(
        f"{output_path.stem}_{args.method}{extra}{output_path.suffix}"
    )

    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    print(f"Pair: {combination}")
    print(f"Loading calibration embeddings: {calib_path}")
    X_calib, calib_paths = load_embeddings_and_paths(calib_path)
    print(f"  shape: {X_calib.shape}, paths: {len(calib_paths)}")

    print(f"Loading test embeddings: {test_path}")
    X_test = load_embeddings(test_path)
    print(f"  shape: {X_test.shape}")

    if X_calib.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: calib D={X_calib.shape[1]}, test D={X_test.shape[1]}"
        )

    n_calib = X_calib.shape[0]
    gamma = _parse_gamma(args.gamma, X_calib, X_test)
    gamma_msg = "sklearn 1/n_features" if gamma is None else f"{gamma:.6g}"
    if args.method == "standard":
        print(
            f"Running standard KMM (B={args.B}, epsilon={args.epsilon}, "
            f"gamma={gamma_msg}) ..."
        )
        weights = get_kmm_weights(
            X_calib, X_test, B=args.B, epsilon=args.epsilon, gamma=gamma
        )
    else:
        print(
            f"Running selective KMM (B={args.B}, epsilon={args.epsilon}, "
            f"tau={args.tau}, gamma={gamma_msg}) ..."
        )
        weights = get_selective_kmm_weights(
            X_calib,
            X_test,
            B=args.B,
            epsilon=args.epsilon,
            tau=args.tau,
            gamma=gamma,
        )

    n_neg = int((weights < 0).sum())
    if n_neg:
        print(
            f"  clipping {n_neg} negative KMM weights to 0 "
            f"(min was {float(weights.min()):.6f})"
        )
    weights = finalize_wcp_weights(
        weights,
        n_calib,
        w_max=float(args.B),
        epsilon=float(args.epsilon),
    )
    weights = np.asarray(weights, dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        w=weights,
        paths=calib_paths,
        calib_npz=str(calib_path),
        test_npz=str(test_path),
        method=np.array(args.method),
        B=np.float32(args.B),
        epsilon=np.float32(args.epsilon),
        tau=np.float32(args.tau) if args.method == "selective" else np.float32(0.0),
        gamma=np.float32(-1.0 if gamma is None else gamma),
        gamma_spec=np.array(str(args.gamma)),
    )

    print(f"Saved KMM weights to: {output_path}")
    print(
        f"  w: min={weights.min():.6f}, max={weights.max():.6f}, "
        f"mean={weights.mean():.6f}, sum={weights.sum():.6f}"
    )


if __name__ == "__main__":
    main()
