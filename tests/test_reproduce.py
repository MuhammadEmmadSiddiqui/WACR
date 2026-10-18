#!/usr/bin/env python3
"""Regenerate each in-scope paper item and compare with results/expected.

Each experiment must recompute from embeddings, copied functions, or
cache_share/ (--from-scores). This test must not compare a cached CSV with itself.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXP = REPO / "experiments"
OUT = REPO / "outputs"
EXP_DIR = REPO / "results" / "expected"
CACHE = REPO / "cache"
CACHE_HIDDEN = REPO / "cache._hidden_from_scores"

ITEMS = [
    ("table1_datasets.csv", "02_build_galleries_scores.py"),
    ("t1_marginal.csv", "03_main_results.py"),
    ("a_main_alpha_sweep.csv", "03_main_results.py"),
    ("table3_robustness.csv", "04_robustness.py"),
    ("fig4_sensitivity.csv", "04_robustness.py"),
    ("table5_rank_budget.csv", "05_pac.py"),
    ("fig6_adaptivity.csv", "06_adaptivity.py"),
    ("table6_weight_quality.csv", "07_weight_quality.py"),
    ("fig7_weight_quality.csv", "07_weight_quality.py"),
    ("table7_conditional.csv", "08_conditional_shift.py"),
    ("table8_ablation.csv", "09_ablation.py"),
    ("table9_sensitivity.csv", "10_sensitivity.py"),
]
FIGURES_AFTER = {
    "03_main_results.py": ("B.png", "A_main.png"),
    "04_robustness.py": ("fig_sensitivity.png", "fig_sensitivity.pdf"),
    "06_adaptivity.py": ("fig_adaptivity.png", "fig_adaptivity.pdf"),
    "07_weight_quality.py": ("fig_weight_quality.png", "fig_weight_quality.pdf"),
    "08_conditional_shift.py": ("fig_conditional.png", "fig_conditional.pdf"),
}


def is_number(s: str) -> bool:
    if s is None or s == "":
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def compare_csv(got: Path, exp: Path, item: str) -> list[str]:
    fails = []
    if got.resolve() == exp.resolve():
        return [f"{item}: regenerated path is the expected file (self-compare)"]
    if not got.is_file():
        return [f"{item}: missing regenerated file {got}"]
    if not exp.is_file():
        return [f"{item}: missing expected file {exp}"]
    g = list(csv.DictReader(got.open(encoding="utf-8")))
    e = list(csv.DictReader(exp.open(encoding="utf-8")))
    if len(g) != len(e):
        fails.append(f"{item}: row count {len(g)} vs expected {len(e)}")
        n = min(len(g), len(e))
        g, e = g[:n], e[:n]
    keys = list(e[0].keys()) if e else []
    for i, (er, gr) in enumerate(zip(e, g)):
        for k in keys:
            ev, gv = er.get(k, ""), gr.get(k, "")
            if is_number(ev) and is_number(gv):
                a, b = float(ev), float(gv)
                if not math.isfinite(a) and not math.isfinite(b):
                    continue
                if abs(a - b) > 1e-6:
                    fails.append(
                        f"{item} row={i+2} col={k} expected={ev} got={gv}"
                    )
                    return fails
            else:
                if str(ev) != str(gv):
                    fails.append(
                        f"{item} row={i+2} col={k} expected={ev!r} got={gv!r}"
                    )
                    return fails
    return fails


def from_scores_requested() -> bool:
    if "--from-scores" in sys.argv:
        return True
    v = os.environ.get("WACR_FROM_SCORES", "").strip().lower()
    return v in {"1", "true", "yes"}


def hide_embedding_cache() -> Path | None:
    if not CACHE.exists():
        return None
    if CACHE_HIDDEN.exists():
        raise SystemExit(f"refuse to hide cache: {CACHE_HIDDEN} already exists")
    CACHE.rename(CACHE_HIDDEN)
    return CACHE_HIDDEN


def restore_embedding_cache(hidden: Path | None) -> None:
    if hidden is None:
        return
    if hidden.exists() and not CACHE.exists():
        hidden.rename(CACHE)


def main() -> None:
    scores = from_scores_requested()
    env = os.environ.copy()
    hidden = None
    if scores:
        from wacr.score_cache import SHARE_DIR, enable, pack_path

        enable()
        env["WACR_FROM_SCORES"] = "1"
        missing = [pack_path(c) for c in ("tinyface_scface", "xqlfw_scface") if not pack_path(c).is_file()]
        if missing:
            print("FAIL")
            print(f"from-scores missing {missing} under {SHARE_DIR}")
            sys.exit(1)
        hidden = hide_embedding_cache()
        print("from-scores: embedding cache hidden; using cache_share/ only", flush=True)
    try:
        ran = set()
        fails = []
        for item, script in ITEMS:
            if script not in ran:
                print(f"=== regenerate {script} ===", flush=True)
                t0 = time.time()
                subprocess.run(
                    [sys.executable, str(EXP / script)],
                    cwd=str(REPO),
                    check=True,
                    env=env,
                )
                ran.add(script)
                for name in FIGURES_AFTER.get(script, ()):
                    fig = REPO / "results" / "figures" / name
                    copied = OUT / name
                    if not fig.is_file():
                        fails.append(f"{name}: missing {fig}")
                    elif fig.stat().st_mtime < t0 - 1:
                        fails.append(f"{name}: {fig} was not redrawn by {script}")
                    if not copied.is_file():
                        fails.append(f"{name}: {script} did not write {copied}")
            fails.extend(compare_csv(OUT / item, EXP_DIR / item, item))
            if fails:
                print("FAIL")
                for line in fails:
                    print(line)
                sys.exit(1)
            print(f"OK {item}")
        print("all items match expected")
    finally:
        restore_embedding_cache(hidden)


if __name__ == "__main__":
    main()
