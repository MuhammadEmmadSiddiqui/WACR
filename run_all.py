#!/usr/bin/env python3
"""Run paper experiments from cached embeddings (default) or from images."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
EXPS = [
    "02_build_galleries_scores.py",
    "03_main_results.py",
    "04_robustness.py",
    "05_pac.py",
    "06_adaptivity.py",
    "07_weight_quality.py",
    "08_conditional_shift.py",
    "09_ablation.py",
    "10_sensitivity.py",
]


def run(script: str) -> None:
    path = REPO / "experiments" / script
    print(f"=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=str(REPO), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from-images", action="store_true")
    p.add_argument(
        "--from-scores",
        action="store_true",
        help="Recompute paper items from cache_share/ only (no embeddings, no images).",
    )
    args = p.parse_args()
    if args.from_images and args.from_scores:
        raise SystemExit("use only one of --from-images / --from-scores")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if args.from_scores:
        from wacr.score_cache import enable

        enable()
    if args.from_images:
        run("01_extract_embeddings.py")
    for s in EXPS:
        run(s)
    print("run_all OK", flush=True)


if __name__ == "__main__":
    main()
