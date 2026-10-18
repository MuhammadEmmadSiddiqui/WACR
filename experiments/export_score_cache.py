#!/usr/bin/env python3
"""Write cache_share/ from embedding caches and fitted baseline weights."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baselines.t1 import fit_baseline_weights
from experiments._common import load_jobs
from wacr.score_cache import SHARE_DIR, dump_job, using_scores


def main() -> None:
    if using_scores():
        raise SystemExit("export_score_cache needs embedding caches; unset WACR_FROM_SCORES")
    wacr, jobs = load_jobs()
    fit_baseline_weights(wacr, jobs)
    paths = [dump_job(job) for job in jobs]
    total = sum(p.stat().st_size for p in paths)
    print(f"cache_share total {total} bytes ({total / (1024 * 1024):.2f} MiB) in {SHARE_DIR}", flush=True)


if __name__ == "__main__":
    main()
