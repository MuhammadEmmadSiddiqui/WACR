#!/usr/bin/env python3
"""tinyface_scface: gallery-matched calibration x every weight vs KMM selective.

Protocol: conformal tau, alpha_cal=alpha, min|C|=0, Phi=top1, beta=0.5.

Two calibration score protocols (test evaluation is identical):
  full    - status-quo WACR scores (top-1 over the full TinyFace gallery)
  gmatch  - scores from TinyFace_Queries_gmatch.npz (top-1 over a 65-id subset)

Method A scores (1 - cos_true) do not depend on the impostor gallery, so Method A
is reported once per weight.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wacr.paths import pair_cache_dir
from wacr.weights import importance_weight_cap, load_importance_weights

SCRIPT_PATH = None  # load_wacr imports wacr.engine
ALPHAS = (0.10, 0.20, 0.25, 0.30)
BETA = 0.5
PREFERRED = (
    "uniform",
    "kmm_selective",
    "kmm_standard",
    "wx",
    "kl",
    "kliep",
    "rulsif",
    "dqm_prox",
    "covtilt",
    "top1lr",
)


def load_wacr():
    from wacr import engine as wacr_mod
    return wacr_mod

def _load_wacr_legacy():
    spec = importlib.util.spec_from_file_location("wacr_mod", SCRIPT_PATH or Path())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_weights(cache: Path, cal: str) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for p in sorted(cache.glob(f"{cal}_Queries_*.npz")):
        tag = p.name[len(f"{cal}_Queries_") : -4]
        if tag in {"mcd", "gmatch"}:
            continue
        try:
            z = np.load(p, allow_pickle=True)
        except OSError:
            continue
        if "w" not in z.files:
            continue
        found[tag] = p
    ordered = [(t, found[t]) for t in PREFERRED if t in found]
    ordered += [(t, found[t]) for t in sorted(found) if t not in PREFERRED]
    return ordered


def prepare_shared(
    wacr, cache: Path, cal: str, tst: str, *, uncertainty_method: str = "top1"
) -> dict:
    calib = wacr.load_bundle(cache / f"{cal}_Queries_mcd.npz", name=f"{cal}_Queries")
    test = wacr.load_bundle(cache / f"{tst}_Queries_mcd.npz", name=f"{tst}_Queries")
    cg = wacr.load_bundle(cache / f"{cal}_Gallery_mcd.npz", name=f"{cal}_Gallery")
    tg = wacr.load_bundle(cache / f"{tst}_Gallery_mcd.npz", name=f"{tst}_Gallery")
    true_calib = wacr.true_match_indices(calib, cg)
    true_test = wacr.true_match_indices(test, tg)
    valid = true_calib >= 0
    if not np.all(valid):
        calib = replace(
            calib,
            embeddings=calib.embeddings[valid],
            paths=[p for p, v in zip(calib.paths, valid) if v],
            fu=(calib.fu[valid] if calib.fu is not None else None),
        )
        true_calib = true_calib[valid]
    calib_sims = wacr.cosine_similarity_matrix(calib.embeddings, cg.embeddings)
    test_sims = wacr.cosine_similarity_matrix(test.embeddings, tg.embeddings)
    s_base = (1.0 - calib_sims[np.arange(calib_sims.shape[0]), true_calib]).astype(
        np.float64
    )
    phi_cal, phi_test = wacr._fit_phi(
        uncertainty_method=uncertainty_method,
        beta=BETA,
        calib=calib,
        test=test,
        calib_sims=calib_sims,
        test_sims=test_sims,
    )
    return {
        "calib_paths": list(calib.paths),
        "s_base": s_base,
        "s_full": s_base / phi_cal,
        "test_sims": test_sims,
        "true_test": true_test,
        "phi_test": phi_test,
        "n_eval": int((true_test >= 0).sum()),
        "n_calib": int(s_base.shape[0]),
    }


def align_gmatch(gmatch_path: Path, calib_paths: list[str]) -> np.ndarray:
    z = np.load(gmatch_path, allow_pickle=True)
    if "s_ad_matched" not in z.files:
        raise SystemExit(f"{gmatch_path}: missing s_ad_matched")
    scores = np.asarray(z["s_ad_matched"], dtype=np.float64).reshape(-1)
    gpaths = [str(p) for p in z["paths"]]
    if gpaths == calib_paths:
        return scores
    idx = {p: i for i, p in enumerate(gpaths)}
    try:
        return np.array([scores[idx[p]] for p in calib_paths], dtype=np.float64)
    except KeyError as exc:
        raise SystemExit(f"{gmatch_path}: missing path {exc}") from None


def eval_one(
    wacr, shared, scores, weights, alpha: float, adaptive: bool, w_test: float | None = None
):
    tau, _ = wacr._fit_threshold(
        scores, weights, alpha=alpha, risk_bound="conformal", delta=0.1, w_test=w_test
    )
    if adaptive:
        res = wacr.evaluate_adaptive(
            shared["test_sims"],
            tau,
            shared["phi_test"],
            shared["true_test"],
            min_set_size=0,
        )
    else:
        res = wacr.evaluate_baseline(
            shared["test_sims"], tau, shared["true_test"], min_set_size=0
        )
    n = shared["n_eval"]
    miss = int(round(float(res.r_hat) * n))
    return float(tau), float(res.r_hat), float(res.avg_set_size), miss


def win_vs_sel(rows, sel_rows) -> tuple[bool, str]:
    """rows/sel_rows: dict alpha -> (tau, rhat, size, miss)."""
    w1_ok = True
    bits = []
    for a in (0.10, 0.20):
        r, c = rows[a][1], rows[a][2]
        cs = sel_rows[a][2]
        ctrl = r <= a
        size_ok = c <= 1.15 * cs
        bits.append(f"W1 a={a:.2f} ctrl={ctrl} |C|={c:.2f}<={1.15*cs:.2f}->{size_ok}")
        w1_ok = w1_ok and ctrl and size_ok
    r25, c25 = rows[0.25][1], rows[0.25][2]
    r30, c30 = rows[0.30][1], rows[0.30][2]
    w2 = r25 <= 0.25
    w3a = c25 < sel_rows[0.25][2]
    w3b = (r30 <= 0.30) and (c30 <= 20.0)
    bits.append(f"W2={w2} R={r25:.4f}")
    bits.append(f"W3a |C|={c25:.2f}<sel {sel_rows[0.25][2]:.2f}->{w3a}")
    bits.append(f"W3b ctrl30={r30<=0.30} |C|={c30:.2f}<=20->{w3b}")
    win = w1_ok and w2 and (w3a or w3b)
    return win, "; ".join(bits)


def print_table(title: str, by_tag: dict, n: int) -> None:
    print()
    print(title)
    hdr = f"{'weight':<16} {'a':>5} {'R-hat':>8} {'|C|':>8} {'miss':>6} {'bar':>5} {'tau':>8}  ok"
    print(hdr)
    print("-" * len(hdr))
    for tag, rows in by_tag.items():
        for a in ALPHAS:
            tau, r, c, m = rows[a]
            bar = int(np.floor(a * n + 1e-12))
            print(
                f"{tag:<16} {a:5.2f} {r:8.4f} {c:8.2f} {m:6d} {bar:5d} {tau:8.4f}  "
                f"{'OK' if r <= a else 'FAIL'}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--combination", default="tinyface_scface")
    p.add_argument("--cal-dataset", default="TinyFace")
    p.add_argument("--test-dataset", default="SCFace")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    comb = normalize_pair_name(args.combination)
    cache = pair_cache_dir(comb)
    cal, tst = args.cal_dataset, args.test_dataset
    gpath = cache / f"{cal}_Queries_gmatch.npz"
    if not gpath.is_file():
        raise SystemExit(f"missing {gpath}")

    wacr = load_wacr()
    shared = prepare_shared(wacr, cache, cal, tst)
    n = shared["n_eval"]
    s_g = align_gmatch(gpath, shared["calib_paths"])
    zg = np.load(gpath, allow_pickle=True)
    print(f"Protocol: {comb} conformal alpha_cal=alpha min|C|=0 top1 beta={BETA}")
    print(f"n_calib={shared['n_calib']} n_eval={n}")
    print(
        f"gmatch: n_gallery_ids={int(zg['n_gallery_ids'])} seed={int(zg['seed'])} "
        f"rank1 full={float(zg['rank1_full']):.3f} matched={float(zg['rank1_matched']):.3f}"
    )

    weights = discover_weights(cache, cal)
    print("weights:", ", ".join(t for t, _ in weights))

    full_rows: dict[str, dict] = {}
    gmatch_rows: dict[str, dict] = {}
    a_rows: dict[str, dict] = {}
    for tag, wpath in weights:
        w = load_importance_weights(wpath, shared["calib_paths"])
        B = importance_weight_cap(wpath, w)
        ess = float(w.sum() ** 2 / max(float((w ** 2).sum()), 1e-12))
        print(f"  {tag}: ess={ess:.0f} min={w.min():.3f} max={w.max():.3f} B={B:.1f}")
        full_rows[tag] = {
            a: eval_one(wacr, shared, shared["s_full"], w, a, True, B) for a in ALPHAS
        }
        gmatch_rows[tag] = {
            a: eval_one(wacr, shared, s_g, w, a, True, B) for a in ALPHAS
        }
        a_rows[tag] = {
            a: eval_one(wacr, shared, shared["s_base"], w, a, False, B) for a in ALPHAS
        }

    print_table("Method B  FULL gallery (status quo)", full_rows, n)
    print_table("Method B  GALLERY-MATCHED calibration", gmatch_rows, n)
    print_table("Method A  (scores ignore gallery matching)", a_rows, n)

    sel = full_rows.get("kmm_selective")
    if sel is None:
        raise SystemExit("kmm_selective weights missing; cannot score WIN rules")

    print()
    print("WIN vs KMM selective (full gallery, Method B)")
    print(f"{'protocol':<8} {'weight':<16} verdict")
    wins = []
    for proto, store in (("full", full_rows), ("gmatch", gmatch_rows)):
        for tag, rows in store.items():
            if proto == "full" and tag == "kmm_selective":
                print(f"{proto:<8} {tag:<16} BASELINE")
                continue
            ok, bits = win_vs_sel(rows, sel)
            print(f"{proto:<8} {tag:<16} {'WIN' if ok else 'LOSS'}")
            if ok:
                wins.append((proto, tag, rows, bits))

    print()
    print("Among gallery-matched Method B, control at all four alphas:")
    controlled = []
    for tag, rows in gmatch_rows.items():
        ok4 = all(rows[a][1] <= a for a in ALPHAS)
        c25, c30 = rows[0.25][2], rows[0.30][2]
        m25, m30 = rows[0.25][3], rows[0.30][3]
        mark = "YES" if ok4 else "no "
        print(
            f"  {mark} {tag:<16} a0.25 miss={m25:3d} |C|={c25:6.2f}   "
            f"a0.30 miss={m30:3d} |C|={c30:6.2f}"
        )
        if ok4:
            vs_sel, _ = win_vs_sel(rows, sel)
            controlled.append((c25, c30, tag, vs_sel, rows))
    if controlled:
        controlled.sort()
        best = controlled[0]
        print()
        print(
            f"WINNING WEIGHT on gallery-matched: {best[2]}  "
            f"(smallest |C| at 0.25 among 4-alpha control; "
            f"WIN-vs-selective={best[3]})"
        )
        r = best[4]
        print(
            f"  vs selective Method B: "
            f"a0.10 {r[0.10][3]} vs {sel[0.10][3]} misses, "
            f"a0.20 {r[0.20][3]} vs {sel[0.20][3]}, "
            f"a0.25 {r[0.25][3]} vs {sel[0.25][3]}, "
            f"a0.30 {r[0.30][3]} vs {sel[0.30][3]}"
        )
    else:
        print("No gallery-matched weight controlled all four alphas.")

    if wins:
        print()
        print("All WIN-vs-selective pairs:")
        for proto, tag, rows, bits in wins:
            print(f"  {proto} + {tag}: {bits}")


if __name__ == "__main__":
    main()
