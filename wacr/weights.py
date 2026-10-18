"""Importance-weight finalization (write) and loading (read).

``finalize_wcp_weights`` is the single WCP normalization primitive. Every
``Inference/compute_*_weights.py`` script calls it **before saving** a ``.npz``
cache: clip negatives, optionally cap at ``w_max``, rescale so ``sum(w) ≈ |Q'|``.

``load_importance_weights`` is the shared **read** path for WACR and W-RCR. It
aligns cached weights to calibration query order and validates them; it does
**not** re-run finalization on standard ``.npz`` files (that work is already
done at write time). Downstream method-specific transforms happen after load.

Write step (``Inference/`` only)
----------------------------------
All weight methods (wx, KMM, KL, KLIEP, RuLSIF, …) call ``finalize_wcp_weights``
before ``np.savez_compressed``. Saved ``.npz`` archives are WCP-ready.

Load step (WACR and W-RCR)
--------------------------
``load_importance_weights``:

- Reorders ``w`` to match calibration query ``paths`` when the cache stores its
  own ``paths`` key.
- Returns ``.npz`` weights as written (no clip/cap/rescale round-trip).
- Runs ``finalize_wcp_weights`` only for legacy ``.npy`` vectors or ``.npz``
  files that still contain negative entries.
- Validates ``sum(w) > 0``.

WACR (Weighted Adaptive Conformal Retrieval)
--------------------------------------------
After load, WACR may drop weights for calibration queries with no gallery
match, then normalizes only inside ``weighted_conformal_threshold``::

    w_norm_i = w_i / sum(w)

Only relative weights affect ``tau``; multiplying all ``w`` by a constant
does not change the threshold. ESS is logged for diagnostics only.

W-RCR (Weighted Risk-Controlling Rank Retrieval)
------------------------------------------------
After load, W-RCR rescales to ``w_tilde`` in ``precompute_calibration``::

    w_tilde_i = |Q'| * w_i / sum(w)     # sum(w_tilde) = |Q'|, mean = 1

Relative weights match WACR's ``w_norm``, but the scale is fixed at ``|Q'|``
for the Empirical Bernstein bound. Optional ``--w-tilde-cap`` water-fills
large entries while keeping ``sum(w_tilde) = |Q'|``, shrinking ``w_tilde_max``
and the bias term in the bound.

Neither method writes a modified weights file; all transforms are in-memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def finalize_wcp_weights(
    w_raw: np.ndarray,
    n_calib: int,
    *,
    w_max: float | None = None,
    epsilon: float = 0.1,
    max_iter: int = 100,
) -> np.ndarray:
    """Make raw density-ratio weights valid for weighted conformal prediction.

    Called by ``Inference/compute_*_weights.py`` before writing ``.npz`` caches.
    Also used on load for legacy ``.npy`` files or damaged ``.npz`` entries with
    negatives (see ``load_importance_weights``).

    Enforces WCP calibration mass: ``(1/n_calib) * sum_i w_i = 1``  ⟺
    ``sum(w) = n_calib``.

    When ``w_max`` is set, uses iterative water-filling (clip capped entries,
    rescale free entries) so neither the sum nor the per-weight cap is violated.
    When ``w_max`` is None, applies a single proportional rescale after clipping
    negatives to zero.
    """
    w = np.clip(np.asarray(w_raw, dtype=np.float64).reshape(-1), 0.0, None)
    if w_max is not None:
        w = np.clip(w, 0.0, float(w_max))

    w_sum = float(w.sum())
    if w_sum <= 0.0:
        raise RuntimeError(
            "All importance weights are zero after clipping; cannot normalize for WCP."
        )

    target_sum = float(n_calib)
    if w_max is None:
        return w * (target_sum / w_sum)

    n = w.size
    if float(w_max) * n <= target_sum:
        return np.full(n, float(w_max), dtype=np.float64)

    for _ in range(max_iter):
        if float(w.max()) <= float(w_max) + 1e-12 and abs(w_sum - target_sum) <= epsilon:
            return w

        capped = w >= float(w_max) - 1e-12
        free_sum = float(w[~capped].sum())
        if free_sum <= 0.0:
            break

        budget = target_sum - float(w_max) * int(capped.sum())
        if budget <= 0.0:
            break

        w[~capped] *= budget / free_sum
        overflow = w > float(w_max) + 1e-12
        if overflow.any():
            w[overflow] = float(w_max)
        else:
            w_sum = float(w.sum())
            if abs(w_sum - target_sum) <= epsilon:
                return w

        w_sum = float(w.sum())

    w = np.clip(w, 0.0, float(w_max))
    w_sum = float(w.sum())
    if w_sum > 0.0 and abs(w_sum - target_sum) > epsilon:
        free = w < float(w_max) - 1e-12
        if free.any() and float(w[free].sum()) > 0.0:
            budget = target_sum - float(w_max) * int((~free).sum())
            if budget > 0.0:
                w[free] *= budget / float(w[free].sum())
                w = np.clip(w, 0.0, float(w_max))

    return w


def decode_paths(arr: np.ndarray) -> List[str]:
    """Decode NPZ ``paths`` arrays that may store bytes or str."""
    out: List[str] = []
    for p in np.asarray(arr).ravel():
        out.append(p.decode("utf-8") if isinstance(p, bytes) else str(p))
    return out


def load_importance_weights(path: Path, expected_paths: List[str]) -> np.ndarray:
    """Load calibration importance weights aligned to ``expected_paths``.

    File formats:

    - ``.npz`` with key ``w`` and optional ``paths`` (reordered to match queries).
      Weights are expected to be WCP-finalized already (written by ``Inference/``).
    - Legacy ``.npy`` vector with no ``paths`` key (WACR only); finalized here.

    Returns non-negative weights with ``sum(w) ≈ |Q'|``. Standard ``.npz`` caches
    are returned without re-finalization. See the module docstring for WACR/W-RCR
    transforms applied after this load step.
    """
    is_legacy_npy = path.suffix.lower() == ".npy"
    w_max: float | None = None
    if is_legacy_npy:
        w_raw = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    else:
        z = np.load(path, allow_pickle=True)
        if "w" not in z.files:
            raise KeyError(f"{path}: missing 'w'; have {list(z.files)}")
        w_raw = np.asarray(z["w"], dtype=np.float64).reshape(-1)
        # Cap metadata: only needed if we must re-finalize (negative entries).
        if "w_max" in z.files:
            w_max = float(z["w_max"])
        elif "B" in z.files:
            w_max = float(z["B"])  # KMM stores the cap as ``B``.

        if "paths" in z.files:
            weight_paths = decode_paths(z["paths"])
            if weight_paths != expected_paths:
                idx = {p: i for i, p in enumerate(weight_paths)}
                if len(idx) != len(weight_paths):
                    raise RuntimeError(f"{path}: duplicate paths in weights cache")
                try:
                    w_raw = np.array(
                        [w_raw[idx[p]] for p in expected_paths], dtype=np.float64
                    )
                except KeyError as exc:
                    raise RuntimeError(
                        f"{path}: weights missing entry for calibration path {exc}"
                    ) from None

    if w_raw.shape[0] != len(expected_paths):
        raise ValueError(
            f"{path}: weights length {w_raw.shape[0]} != |Q'|={len(expected_paths)}"
        )

    n_neg = int((w_raw < 0).sum())
    if n_neg:
        neg_min = float(w_raw.min())
        print(
            f"  [Weights] clipping {n_neg} negative values to 0 (min was {neg_min:.3e})"
        )

    n_calib = len(expected_paths)
    if is_legacy_npy or n_neg:
        # Legacy .npy has no write-time finalization; negatives need repair + rescale.
        w = finalize_wcp_weights(
            w_raw,
            n_calib,
            w_max=w_max,
            epsilon=0.1,
        )
    else:
        # Standard .npz from Inference: trust stored weights (already finalized).
        w = np.clip(np.asarray(w_raw, dtype=np.float64).reshape(-1), 0.0, None)

    if float(w.sum()) <= 0.0:
        raise ValueError(f"{path}: sum(weights) must be > 0; got {float(w.sum()):.6f}")
    return w


def importance_weight_cap(path: Path, weights: np.ndarray | None = None) -> float:
    """Upper bound ``B`` on any importance weight, for the WCP test atom.

    Weighted split-CP needs a test-point weight that upper-bounds the unknown
    ``w(x_test)``. Every ``Inference/compute_*_weights.py`` caps weights at
    ``w_max`` (or ``B`` for KMM) during finalization, so that cap is a valid
    bound for the test point as well. Files without a cap (e.g. ``uniform``)
    fall back to ``max(w)``.
    """
    if path.suffix.lower() != ".npy":
        z = np.load(path, allow_pickle=True)
        if "w_max" in z.files:
            return float(z["w_max"])
        if "B" in z.files:
            return float(z["B"])
    if weights is None:
        raise ValueError(f"{path}: no cap stored and no weights given for fallback")
    return float(np.max(np.asarray(weights, dtype=np.float64)))


def effective_sample_size(w: np.ndarray, *, eps: float = 1e-12) -> float:
    """Effective sample size for non-negative importance weights.

    ``ESS = (sum w)^2 / (sum w^2 + eps)``. Returns ``nan`` when ``sum(w) <= 0``.
    Negative entries are clipped to 0 first (diagnostic helper; load clips only
    when negatives are present, and standard ``.npz`` caches should have none).
    """
    w = np.clip(np.asarray(w, dtype=np.float64).reshape(-1), 0.0, None)
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return float("nan")
    return (w_sum ** 2) / float((w ** 2).sum() + eps)
