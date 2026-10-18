"""RA-FR rank-budget transfer (percentile / count)."""

from __future__ import annotations

import math

import numpy as np

from wacr.protocol import ALPHAS, BETA

SEED = 0


def matched_rank_and_t(calib_sims, calib_ids, gallery_ids, *, n_gallery_ids, seed):
    rng = np.random.default_rng(seed)
    gallery_ids = np.asarray(gallery_ids)
    by_id: dict[str, list[int]] = {}
    for i, ident in enumerate(gallery_ids):
        by_id.setdefault(str(ident), []).append(i)
    by_id = {k: np.asarray(v, dtype=np.int64) for k, v in by_id.items()}
    id_list = np.array(sorted(by_id.keys()))
    n_imp = int(max(0, min(n_gallery_ids - 1, len(id_list) - 1)))
    n = calib_sims.shape[0]
    top1 = np.empty(n, dtype=np.float64)
    rank = np.empty(n, dtype=np.int64)
    n_gal = np.empty(n, dtype=np.int64)
    gids = np.asarray([str(x) for x in gallery_ids])
    for i in range(n):
        own_id = str(calib_ids[i])
        own = by_id[own_id]
        others = id_list[id_list != own_id]
        pick = rng.choice(others, size=n_imp, replace=False)
        imp = np.concatenate([by_id[p] for p in pick]) if n_imp > 0 else np.empty(0, np.int64)
        cand = np.concatenate([own, imp])
        sims = calib_sims[i, cand]
        top1[i] = float(sims.max())
        n_gal[i] = int(cand.size)
        order = np.argsort(-sims, kind="mergesort")
        hit = np.flatnonzero(gids[cand][order] == own_id)
        rank[i] = int(hit[0]) + 1
    return top1, rank, n_gal


def test_ranks(test_sims, true_idx):
    n = test_sims.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if true_idx[i] < 0:
            continue
        order = np.argsort(-test_sims[i], kind="mergesort")
        out[i] = int(np.flatnonzero(order == true_idx[i])[0]) + 1
    return out


def k_cut(kappa, phi, n_gal, n_te, mode: str):
    k_test = np.minimum(np.maximum(np.ceil(kappa * phi), 1).astype(np.int64), n_te)
    if mode == "count":
        return np.minimum(k_test, n_gal), k_test
    k_cal = np.ceil(k_test.astype(np.float64) * n_gal / float(n_te)).astype(np.int64)
    k_cal = np.minimum(n_gal, np.maximum(1, k_cal))
    return k_cal, k_test
