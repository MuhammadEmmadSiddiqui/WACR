#!/usr/bin/env python3
"""Build gallery-matched scores (if missing) and Table 1 Recall@K."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments._common import load_cfg, load_jobs, out_dir, write_csv
from wacr.score_cache import using_scores

KS = (1, 5, 10)
SEED = 0
K_IDS = 65


def recall_full(sims: np.ndarray, qids: list[str], gids: list[str], ks=KS) -> dict:
    qids_a = np.asarray([str(x) for x in qids])
    gids_a = np.asarray([str(x) for x in gids])
    order = np.argsort(-sims, axis=1, kind="mergesort")
    n = int(sims.shape[0])
    rec = {}
    for k in ks:
        kk = int(k)
        top = gids_a[order[:, :kk]]
        h = np.any(top == qids_a[:, None], axis=1)
        rec[kk] = float(h.mean())
    return {"recall": rec, "n": n}


def recall_matched(sims, qids, gids, *, n_gallery_ids=K_IDS, seed=SEED, ks=KS) -> dict:
    rng = np.random.default_rng(seed)
    gallery_ids = np.asarray([str(x) for x in gids])
    calib_ids = np.asarray([str(x) for x in qids])
    by_id: dict[str, list[int]] = {}
    for i, ident in enumerate(gallery_ids):
        by_id.setdefault(ident, []).append(i)
    by_id_arr = {k: np.asarray(v, dtype=np.int64) for k, v in by_id.items()}
    id_list = np.array(sorted(by_id_arr.keys()))
    n_imp = int(max(0, min(n_gallery_ids - 1, len(id_list) - 1)))
    n = sims.shape[0]
    hits = {int(k): 0 for k in ks}
    for i in range(n):
        own_id = str(calib_ids[i])
        own = by_id_arr[own_id]
        others = id_list[id_list != own_id]
        pick = rng.choice(others, size=n_imp, replace=False)
        imp = np.concatenate([by_id_arr[p] for p in pick]) if n_imp > 0 else np.empty(0, np.int64)
        cand = np.concatenate([own, imp])
        scores = sims[i, cand]
        order = np.argsort(-scores, kind="mergesort")
        ranked_ids = gallery_ids[cand[order]]
        for k in ks:
            if np.any(ranked_ids[: int(k)] == own_id):
                hits[int(k)] += 1
    return {"recall": {k: hits[k] / float(n) for k in hits}, "n": n}


def main() -> None:
    cfg = load_cfg()
    wacr, jobs = load_jobs()
    print(f"loaded {len(jobs)} jobs k={cfg['k']} from_scores={using_scores()}", flush=True)
    rows = []
    by_cal = {j["cal"]: j for j in jobs}
    for cal in ("TinyFace", "XQLFW"):
        job = by_cal[cal]
        rec = recall_matched(
            job["calib_sims"],
            job["calib_ids"],
            job["cg_ids"],
            n_gallery_ids=int(cfg["k"]),
            seed=int(cfg["seed"]),
        )
        rows.append(
            {
                "dataset": cal,
                "role": "Calibration source",
                "identities": len(set(map(str, job["calib_ids"]))),
                "probes": int(job["n_calib"]),
                "gallery": int(job["calib_sims"].shape[1]),
                "recall_at_1": rec["recall"][1],
                "recall_at_5": rec["recall"][5],
                "recall_at_10": rec["recall"][10],
            }
        )
        print(f"{cal} matched R@1={rec['recall'][1]:.6f} n={rec['n']}", flush=True)

    job = jobs[0]
    rec = recall_full(job["test_sims"], job["test_qids"], job["tg_ids"])
    rows.append(
        {
            "dataset": "SCFace",
            "role": "Deployment target",
            "identities": len(set(map(str, job["tg_ids"]))),
            "probes": int(job["test_sims"].shape[0]),
            "gallery": int(job["test_sims"].shape[1]),
            "recall_at_1": rec["recall"][1],
            "recall_at_5": rec["recall"][5],
            "recall_at_10": rec["recall"][10],
        }
    )
    print(f"SCFace full R@1={rec['recall'][1]:.6f} n={rec['n']}", flush=True)
    write_csv(out_dir() / "table1_datasets.csv", rows)


if __name__ == "__main__":
    main()
