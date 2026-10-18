#!/usr/bin/env python3
"""Gallery-matched calibration for WACR (label-free w.r.t. test queries).

Motivation (tinyface_scface diagnostics): the calibration and test query
embeddings are indistinguishable (domain classifier at chance), so no w(x) has
anything to estimate. The calibration-to-test score gap lives in the retrieval
context instead: the test gallery has far fewer identities (65 vs 1722), so at
equal top-1 similarity the test rank-1 rate is much higher, and test queries sit
farther from every gallery item (lower top-1 similarity).

This module addresses both parts without touching test labels:

1. Gallery matching. Each calibration query is re-scored against a random
   subset of the calibration gallery containing its own identity plus
   ``n_test_ids - 1`` impostor identities, so the impostor structure matches the
   test gallery. Only the test gallery's identity count is used.
2. Retrieval-context weights. A 1-D logistic density ratio on top-1 similarity
   (matched-gallery calib top-1 vs real-gallery test top-1), finalized with the
   same ``w_max=30, epsilon=0.1`` WCP transform as every other weight file.

Outputs (``cache/<COMB>/``):

- ``<CAL>_Queries_gmatch.npz``: per-calibration-query matched top-1, Phi, scores.
- ``<CAL>_Queries_top1lr.npz``: standard weight file (``w``, ``paths``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wacr.embeddings import EmbeddingBundle, load_bundle  # noqa: E402
from wacr.paths import (  # noqa: E402
    DEFAULT_CALIB_DATASET,
    DEFAULT_TEST_DATASET,
    add_combination_argument,
    add_pair_argument,
    combination_cache_dir,
    pair_cache_dir,
    normalize_combination_name,
    normalize_pair_name,
)
from wacr.similarity import cosine_similarity_matrix  # noqa: E402
from wacr.weights import finalize_wcp_weights  # noqa: E402


def gallery_identity_index(gallery: EmbeddingBundle) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = {}
    for i, ident in enumerate(gallery.identities):
        idx.setdefault(ident, []).append(i)
    return idx


def true_match_indices(queries: EmbeddingBundle, gallery: EmbeddingBundle, sims: np.ndarray) -> np.ndarray:
    by_id = gallery_identity_index(gallery)
    out = np.full(len(queries.paths), -1, dtype=np.int64)
    for qi, ident in enumerate(queries.identities):
        cands = by_id.get(ident)
        if not cands:
            continue
        arr = np.asarray(cands, dtype=np.int64)
        out[qi] = int(arr[int(np.argmax(sims[qi, arr]))])
    return out


def matched_gallery_top1(
    calib_sims: np.ndarray,
    calib_ids: Sequence[str],
    gallery_ids: Sequence[str],
    *,
    n_gallery_ids: int,
    seed: int,
) -> np.ndarray:
    """Top-1 similarity of each calibration query over a random gallery subset.

    The subset contains the query's own identity plus ``n_gallery_ids - 1``
    impostor identities (all their images). Returns shape ``(n_calib,)``.
    """
    rng = np.random.default_rng(seed)
    gallery_ids = np.asarray(gallery_ids)
    by_id: Dict[str, np.ndarray] = {}
    for i, ident in enumerate(gallery_ids):
        by_id.setdefault(str(ident), []).append(i)
    by_id = {k: np.asarray(v, dtype=np.int64) for k, v in by_id.items()}
    id_list = np.array(sorted(by_id.keys()))
    n_imp = int(max(0, min(n_gallery_ids - 1, len(id_list) - 1)))

    n = calib_sims.shape[0]
    top1 = np.empty(n, dtype=np.float64)
    for i in range(n):
        own_id = str(calib_ids[i])
        own = by_id[own_id]
        others = id_list[id_list != own_id]
        pick = rng.choice(others, size=n_imp, replace=False)
        imp = np.concatenate([by_id[p] for p in pick]) if n_imp > 0 else np.empty(0, np.int64)
        cand = np.concatenate([own, imp])
        top1[i] = float(calib_sims[i, cand].max())
    return top1


def top1_density_ratio_weights(top1_calib: np.ndarray, top1_test: np.ndarray) -> np.ndarray:
    """w(top1) = p_test(top1) / p_calib(top1) via 1-D logistic regression (raw, unfinalized)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    a = np.asarray(top1_calib, np.float64).reshape(-1, 1)
    b = np.asarray(top1_test, np.float64).reshape(-1, 1)
    X = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X, y)
    p = clf.predict_proba(a)[:, 1]
    return (p / (1.0 - p + 1e-12)) * (len(a) / len(b))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gallery-matched calibration + top-1 ratio weights.")
    add_combination_argument(p)
    p.add_argument("--cal-dataset", default=DEFAULT_CALIB_DATASET)
    p.add_argument("--test-dataset", default=DEFAULT_TEST_DATASET)
    p.add_argument("--beta", type=float, default=0.5, help="Phi = clip(1 - top1, 0, 1) + beta")
    p.add_argument("--n-gallery-ids", type=int, default=None,
                   help="Identities per matched gallery (default: number of identities in the test gallery).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--w-max", type=float, default=30.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    comb = normalize_combination_name(args.combination)
    cache = combination_cache_dir(comb)
    cal, tst = args.cal_dataset, args.test_dataset

    calib = load_bundle(cache / f"{cal}_Queries_mcd.npz", name=f"{cal}_Queries")
    test = load_bundle(cache / f"{tst}_Queries_mcd.npz", name=f"{tst}_Queries")
    cg = load_bundle(cache / f"{cal}_Gallery_mcd.npz", name=f"{cal}_Gallery")
    tg = load_bundle(cache / f"{tst}_Gallery_mcd.npz", name=f"{tst}_Gallery")

    calib_sims = cosine_similarity_matrix(calib.embeddings, cg.embeddings)
    test_sims = cosine_similarity_matrix(test.embeddings, tg.embeddings)
    true_c = true_match_indices(calib, cg, calib_sims)
    if not np.all(true_c >= 0):
        raise RuntimeError(f"{int((true_c < 0).sum())} calibration queries lack a gallery match")

    n_test_ids = len(set(tg.identities))
    n_ids = int(args.n_gallery_ids) if args.n_gallery_ids else n_test_ids
    print(f"Pair: {comb}")
    print(f"  calib gallery: {cg.embeddings.shape[0]} imgs / {len(set(cg.identities))} ids")
    print(f"  test gallery:  {tg.embeddings.shape[0]} imgs / {n_test_ids} ids -> matched gallery uses {n_ids} ids")

    n = calib_sims.shape[0]
    cos_true = calib_sims[np.arange(n), true_c]
    top1_full = calib_sims.max(axis=1)
    top1_m = matched_gallery_top1(
        calib_sims, calib.identities, cg.identities, n_gallery_ids=n_ids, seed=args.seed
    )
    phi_m = np.clip(1.0 - top1_m, 0.0, 1.0) + args.beta
    s_base = 1.0 - cos_true
    s_ad_m = s_base / phi_m
    rank1_full = float(np.mean(cos_true >= top1_full))
    rank1_m = float(np.mean(cos_true >= top1_m))

    top1_t = test_sims.max(axis=1)
    w_raw = top1_density_ratio_weights(top1_m, top1_t)
    w = finalize_wcp_weights(w_raw, n, w_max=args.w_max, epsilon=args.epsilon).astype(np.float32)
    ess = float(w.sum() ** 2 / (w.astype(np.float64) ** 2).sum())

    print(f"  calib rank-1: full gallery {rank1_full:.3f} -> matched {rank1_m:.3f}")
    print(f"  calib top1 mean: full {top1_full.mean():.4f} -> matched {top1_m.mean():.4f}; test {top1_t.mean():.4f}")
    print(f"  w(top1): min={w.min():.3f} max={w.max():.3f} ess={ess:.0f}/{n}")

    out_g = cache / f"{cal}_Queries_gmatch.npz"
    np.savez_compressed(
        out_g,
        paths=np.asarray(calib.paths, dtype=object),
        top1_matched=top1_m.astype(np.float32),
        top1_full=top1_full.astype(np.float32),
        cos_true=cos_true.astype(np.float32),
        phi_matched=phi_m.astype(np.float32),
        s_base=s_base.astype(np.float32),
        s_ad_matched=s_ad_m.astype(np.float32),
        beta=np.float32(args.beta),
        n_gallery_ids=np.int32(n_ids),
        seed=np.int32(args.seed),
        rank1_full=np.float32(rank1_full),
        rank1_matched=np.float32(rank1_m),
    )
    out_w = cache / f"{cal}_Queries_top1lr.npz"
    np.savez_compressed(
        out_w,
        w=w,
        paths=np.asarray(calib.paths, dtype=object),
        method=np.array("top1lr"),
        w_max=np.float32(args.w_max),
        epsilon=np.float32(args.epsilon),
        ess=np.float32(ess),
        n_gallery_ids=np.int32(n_ids),
        seed=np.int32(args.seed),
    )
    print(f"Saved {out_g}")
    print(f"Saved {out_w}")
    print(f"wall={time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
