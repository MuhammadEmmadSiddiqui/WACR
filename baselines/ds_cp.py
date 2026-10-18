#!/usr/bin/env python3
"""DS-CP (Lin et al., ICML 2026) on frozen WACR scores and galleries.

Follows github.com/zhexiaolin/CP commit 60be9df (src/cp_methods.py, src/main-mmlu.py).
Does not overwrite frozen WACR / WRCP / Tab04-05 files.

Prints are ASCII only (Windows cp1252).
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TRAIN = REPO / "Train"
for p in (str(REPO), str(HERE), str(TRAIN)):
    if p not in sys.path:
        sys.path.insert(0, p)

import wacr.protocol as evp  # noqa: E402
from wacr.jobs import load_wacr  # noqa: E402
from wacr.weights import (  # noqa: E402
    decode_paths,
    effective_sample_size,
    load_importance_weights,
)

ALPHAS = (0.05, 0.10, 0.20)
GAMMAS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)
W_MAX = evp.W_MAX
OUT = HERE / "dscp"
CLIP_MIN = 1e-6
CLIP_MAX = 1.0 - 1e-6
XGB_KW = dict(
    max_depth=3,
    n_estimators=50,
    learning_rate=0.2,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=5,
    reg_lambda=10,
    eval_metric="logloss",
    random_state=42,
    n_jobs=1,
)
WCP_DC = {
    ("TinyFace", 0.05): (0.03692307692307695, 68.55076923076923),
    ("TinyFace", 0.10): (0.11076923076923073, 20.09948717948718),
    ("TinyFace", 0.20): (0.25230769230769234, 4.781538461538462),
    ("XQLFW", 0.05): (0.20102564102564102, 7.838974358974359),
    ("XQLFW", 0.10): (0.3056410256410257, 2.9261538461538463),
    ("XQLFW", 0.20): (0.47282051282051285, 1.4461538461538461),
}


def log(msg: str) -> None:
    print(
        msg.replace("\u03b3", "gamma")
        .replace("\u03b1", "alpha")
        .replace("\u2192", "->")
        .replace("\u221e", "inf"),
        flush=True,
    )


def remap_image_path(stored: str) -> Path:
    s = stored.replace("\\", "/")
    marker = "/datasets/"
    if marker in s:
        rel = s.split(marker, 1)[1]
        return (REPO / "datasets" / rel).resolve()
    p = Path(stored)
    if p.is_file():
        return p
    raise FileNotFoundError(stored)


def l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return x / nrm


def density_ratio(emb: np.ndarray, clf) -> np.ndarray:
    """src/cp_methods.py density_ratio, lines 104-107."""
    p = clf.predict_proba(emb)[:, 1]
    p = np.clip(p, CLIP_MIN, CLIP_MAX)
    return p / (1.0 - p + 1e-12)


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, alpha: float, gamma: float) -> float:
    """src/cp_methods.py weighted_quantile, lines 110-129, with +inf for vacuity."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return float("inf")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    w = weights[order]
    csum = np.cumsum(w)
    cutoff = (1.0 - alpha) * (float(w.sum()) + float(gamma) * float(w.max()))
    loc = int(np.searchsorted(csum, cutoff, side="left"))
    if loc >= s.size:
        return float("inf")
    return float(s[loc])


def split_cp_threshold(scores: np.ndarray, alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th smallest score, +inf if that index is past n."""
    s = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1), kind="mergesort")
    n = int(s.size)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def build_classifier(name: str):
    if name == "XGBoost":
        return XGBClassifier(**XGB_KW)
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=42,
        )
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    raise ValueError(name)


def align_fe(wx_path: Path, expected_paths: list[str]) -> np.ndarray:
    z = np.load(wx_path, allow_pickle=True)
    paths = decode_paths(z["paths"])
    fe = np.asarray(z["fe"], dtype=np.float64)
    idx = {p: i for i, p in enumerate(paths)}
    return np.stack([fe[idx[p]] for p in expected_paths], axis=0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wtr.writeheader()
        wtr.writerows(rows)
    log(f"Wrote {path}")


def diagnostics(t_cal: np.ndarray, w: np.ndarray, t_te: np.ndarray) -> dict:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise RuntimeError("weights must be finite and positive")
    ess = float(effective_sample_size(w))
    n = int(w.size)
    t_cut = float(np.median(t_te))
    return {
        "w_min": float(w.min()),
        "w_median": float(np.median(w)),
        "w_max": float(w.max()),
        "ess_over_n": ess / n,
        "tv": evp._tv_hist(t_cal, w, t_te, 20),
        "low_t_gap": evp._tv_low_t(t_cal, w, t_te, t_cut),
    }


def eval_tau(wacr, job, tau: float, min_set_size: int):
    res = wacr.evaluate_adaptive(
        job["test_sims"], tau, job["phi_test"], job["true_test"], min_set_size=min_set_size
    )
    return float(res.r_hat), float(res.avg_set_size)


def load_or_embed_scface(wacr, job: dict) -> np.ndarray:
    """DINOv2 for the valid SCFace queries in job order."""
    import torch
    from train_domain_classifier_embeddings import extract_dinov2_embeddings
    from wacr.embeddings import load_bundle

    test = load_bundle(job["cache"] / "SCFace_Queries_mcd.npz", name="SCFace_Queries")
    gallery = load_bundle(job["cache"] / "SCFace_Gallery_mcd.npz", name="SCFace_Gallery")
    true_all = wacr.true_match_indices(test, gallery)
    keep_idx = np.flatnonzero(true_all >= 0)
    paths_kept = [test.paths[int(i)] for i in keep_idx]
    if len(paths_kept) != int(job["n_eval"]):
        raise RuntimeError(f"SCFace path count {len(paths_kept)} != n_eval {job['n_eval']}")

    out = OUT / f"{job['cal']}_SCFace_Queries_dinov2.npz"
    if out.is_file():
        z = np.load(out, allow_pickle=True)
        cached = decode_paths(z["paths"])
        if cached == paths_kept:
            log(f"  loaded DINOv2 SCFace cache {out}")
            return np.asarray(z["fe"], dtype=np.float64)

    img_paths = [remap_image_path(p) for p in paths_kept]
    missing = [p for p in img_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images, e.g. {missing[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  embedding {len(img_paths)} SCFace queries with DINOv2 on {device} ...")
    fe = extract_dinov2_embeddings(
        image_paths=img_paths,
        hub_name="dinov2_vitb14",
        device=device,
        embed_batch_size=32,
        img_size=112,
        allow_upscale=False,
    )
    fe_np = np.asarray(fe.detach().cpu().numpy(), dtype=np.float64)
    np.savez_compressed(out, paths=np.asarray(paths_kept, dtype=object), fe=fe_np.astype(np.float32))
    log(f"  wrote {out}")
    return fe_np


def fit_weights(x_cal: np.ndarray, x_te: np.ndarray, clf_name: str) -> np.ndarray:
    x_cal = l2_rows(x_cal)
    x_te = l2_rows(x_te)
    x = np.vstack([x_cal, x_te])
    y = np.concatenate([np.zeros(x_cal.shape[0], dtype=np.int32), np.ones(x_te.shape[0], dtype=np.int32)])
    clf = build_classifier(clf_name)
    clf.fit(x, y)
    w = density_ratio(x_cal, clf)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise RuntimeError(f"{clf_name} produced non-positive or non-finite weights")
    return w


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wacr = load_wacr()
    jobs = [evp.load_job(wacr, *spec) for spec in evp.JOBS]

    # --- sanity: equal weights, gamma=1 equals split-CP ---
    for job in jobs:
        scores = job["s_g"]
        w_one = np.ones(scores.shape[0], dtype=np.float64)
        for a in ALPHAS:
            q_ds = weighted_quantile(scores, w_one, a, gamma=1.0)
            q_sp = split_cp_threshold(scores, a)
            if not (
                (not np.isfinite(q_ds) and not np.isfinite(q_sp))
                or (np.isfinite(q_ds) and np.isfinite(q_sp) and abs(q_ds - q_sp) <= 1e-12)
            ):
                raise AssertionError(
                    f"SANITY {job['cal']} a={a:.2f} DS-CP q={q_ds} != split-CP {q_sp}"
                )
        log(f"SANITY OK {job['cal']}: w=1, gamma=1 matches split-CP at all alpha")

    main_rows: list[dict] = []
    diag_rows: list[dict] = []
    ablate_rows: list[dict] = []
    empty_rows: list[dict] = []

    for job in jobs:
        cal = job["cal"]
        wx_path = job["cache"] / f"{cal}_Queries_wx.npz"
        fe_dino_cal = align_fe(wx_path, job["calib_paths"])
        fe_dino_te = load_or_embed_scface(wacr, job)
        fe_ada_cal = job["calib_emb"]
        fe_ada_te = job["test_emb"]
        scores = job["s_g"]
        log(f"=== {cal} -> SCFace  n_cal={job['n_calib']} n_eval={job['n_eval']} ===")

        # Existing WCP-DC (MLP, finalized wx, B-atom 30)
        w_dc = load_importance_weights(wx_path, job["calib_paths"])
        for a in ALPHAS:
            tau = wacr.weighted_conformal_threshold(scores, w_dc, alpha=a, w_test=W_MAX)
            r, c = eval_tau(wacr, job, tau, 0)
            tgt = WCP_DC[(cal, a)]
            if abs(r - tgt[0]) > 5.5e-4 and round(r, 3) != round(tgt[0], 3):
                raise RuntimeError(f"WCP-DC {cal} a={a:.2f} R-hat={r:.6f} expected {tgt[0]}")
            main_rows.append(
                {
                    "cal": cal,
                    "method": "WCP-DC (MLP)",
                    "alpha": a,
                    "r_hat": r,
                    "set_size": c,
                    "tau_inf": int(not np.isfinite(tau)),
                }
            )
            log(f"  WCP-DC (MLP) a={a:.2f}  R-hat={r:.4f}  |C|={c:.1f}")

        # Main DS-CP: XGBoost + DINOv2 + gamma=1
        w_main = fit_weights(fe_dino_cal, fe_dino_te, "XGBoost")
        d = diagnostics(job["t_cal"], w_main, job["t_te"])
        d.update({"cal": cal, "embedding": "DINOv2", "classifier": "XGBoost"})
        diag_rows.append(d)
        log(
            f"  DS-CP weights min={d['w_min']:.4g} med={d['w_median']:.4g} "
            f"max={d['w_max']:.4g} ESS/n={100*d['ess_over_n']:.1f}% "
            f"TV={d['tv']:.4f} low-t={d['low_t_gap']:+.4f}"
        )
        for a in ALPHAS:
            tau = weighted_quantile(scores, w_main, a, gamma=1.0)
            r0, c0 = eval_tau(wacr, job, tau, 0)
            r1, c1 = eval_tau(wacr, job, tau, 1)
            main_rows.append(
                {
                    "cal": cal,
                    "method": "DS-CP (XGBoost, DINOv2, gamma=1)",
                    "alpha": a,
                    "r_hat": r0,
                    "set_size": c0,
                    "tau_inf": int(not np.isfinite(tau)),
                }
            )
            empty_rows.append(
                {
                    "cal": cal,
                    "alpha": a,
                    "r_hat_min0": r0,
                    "set_size_min0": c0,
                    "r_hat_min1": r1,
                    "set_size_min1": c1,
                    "delta_r": r1 - r0,
                    "delta_c": c1 - c0,
                }
            )
            log(
                f"  DS-CP a={a:.2f}  R-hat={r0:.4f}  |C|={c0:.1f}  "
                f"min1 R-hat={r1:.4f} |C|={c1:.1f}  dR={r1-r0:.4g} dC={c1-c0:.4g}"
            )

        # Ablations at alpha=0.10
        a = 0.10
        for g in GAMMAS:
            tau = weighted_quantile(scores, w_main, a, gamma=g)
            r, c = eval_tau(wacr, job, tau, 0)
            ablate_rows.append(
                {
                    "cal": cal,
                    "family": "gamma",
                    "embedding": "DINOv2",
                    "classifier": "XGBoost",
                    "gamma": g,
                    "r_hat": r,
                    "set_size": c,
                    "tau_inf": int(not np.isfinite(tau)),
                }
            )
            log(f"  ablation gamma={g:g}  R-hat={r:.4f}  |C|={c:.1f}")

        for clf_name in ("XGBoost", "MLP", "LogisticRegression"):
            w = w_main if clf_name == "XGBoost" else fit_weights(fe_dino_cal, fe_dino_te, clf_name)
            tau = weighted_quantile(scores, w, a, gamma=1.0)
            r, c = eval_tau(wacr, job, tau, 0)
            dd = diagnostics(job["t_cal"], w, job["t_te"])
            ablate_rows.append(
                {
                    "cal": cal,
                    "family": "classifier",
                    "embedding": "DINOv2",
                    "classifier": clf_name,
                    "gamma": 1.0,
                    "r_hat": r,
                    "set_size": c,
                    "tau_inf": int(not np.isfinite(tau)),
                    "tv": dd["tv"],
                    "ess_over_n": dd["ess_over_n"],
                    "w_min": dd["w_min"],
                    "w_max": dd["w_max"],
                }
            )
            log(f"  ablation clf={clf_name}  R-hat={r:.4f}  |C|={c:.1f}  TV={dd['tv']:.4f}")

        for emb_name, x_cal, x_te in (
            ("DINOv2", fe_dino_cal, fe_dino_te),
            ("AdaFace", fe_ada_cal, fe_ada_te),
        ):
            w = w_main if emb_name == "DINOv2" else fit_weights(x_cal, x_te, "XGBoost")
            tau = weighted_quantile(scores, w, a, gamma=1.0)
            r, c = eval_tau(wacr, job, tau, 0)
            dd = diagnostics(job["t_cal"], w, job["t_te"])
            ablate_rows.append(
                {
                    "cal": cal,
                    "family": "embedding",
                    "embedding": emb_name,
                    "classifier": "XGBoost",
                    "gamma": 1.0,
                    "r_hat": r,
                    "set_size": c,
                    "tau_inf": int(not np.isfinite(tau)),
                    "tv": dd["tv"],
                    "ess_over_n": dd["ess_over_n"],
                    "w_min": dd["w_min"],
                    "w_max": dd["w_max"],
                }
            )
            if emb_name == "AdaFace":
                dd2 = dict(dd)
                dd2.update({"cal": cal, "embedding": "AdaFace", "classifier": "XGBoost"})
                diag_rows.append(dd2)
            log(f"  ablation emb={emb_name}  R-hat={r:.4f}  |C|={c:.1f}  TV={dd['tv']:.4f}")

    _write_csv(OUT / "main.csv", main_rows)
    _write_csv(OUT / "diagnostics.csv", diag_rows)
    _write_csv(OUT / "ablation.csv", ablate_rows)
    _write_csv(OUT / "empty_set.csv", empty_rows)
    write_tex(main_rows, diag_rows, ablate_rows, empty_rows)
    write_diff_note(empty_rows)
    log("OK")


def _fmt_r(x: float) -> str:
    return f"{x:.3f}"


def _fmt_c(x: float) -> str:
    return f"{x:.1f}"


def write_tex(main_rows, diag_rows, ablate_rows, empty_rows) -> None:
    def pick(rows, **kw):
        for r in rows:
            ok = True
            for k, v in kw.items():
                if k in ("alpha", "gamma"):
                    if abs(float(r[k]) - float(v)) >= 1e-12:
                        ok = False
                        break
                elif r.get(k) != v:
                    ok = False
                    break
            if ok:
                return r
        raise KeyError(kw)

    lines = []
    for cal in ("TinyFace", "XQLFW"):
        for a in ALPHAS:
            ds = pick(main_rows, cal=cal, method="DS-CP (XGBoost, DINOv2, gamma=1)", alpha=a)
            dc = pick(main_rows, cal=cal, method="WCP-DC (MLP)", alpha=a)
            lines.append(
                f"{cal} & {a:.2f} & {_fmt_r(ds['r_hat'])} & {_fmt_c(ds['set_size'])} && "
                f"{_fmt_r(dc['r_hat'])} & {_fmt_c(dc['set_size'])} \\\\"
            )
        if cal == "TinyFace":
            lines.append("\\addlinespace")

    dlines = []
    for cal in ("TinyFace", "XQLFW"):
        d = pick(diag_rows, cal=cal, embedding="DINOv2")
        dlines.append(
            f"{cal} & {d['tv']:.3f} & ${d['low_t_gap']:+.3f}$ & "
            f"{100*d['ess_over_n']:.1f}\\% & ${d['w_min']:.2f}$--${d['w_max']:.2f}$ \\\\"
        )

    glines = []
    for cal in ("TinyFace", "XQLFW"):
        for g in GAMMAS:
            r = pick(ablate_rows, cal=cal, family="gamma", gamma=g)
            glines.append(
                f"{cal} & {g:g} & {_fmt_r(r['r_hat'])} & {_fmt_c(r['set_size'])} \\\\"
            )
        if cal == "TinyFace":
            glines.append("\\addlinespace")

    clines = []
    for cal in ("TinyFace", "XQLFW"):
        for name in ("XGBoost", "MLP", "LogisticRegression"):
            r = pick(ablate_rows, cal=cal, family="classifier", classifier=name)
            clines.append(
                f"{cal} & {name} & {_fmt_r(r['r_hat'])} & {_fmt_c(r['set_size'])} \\\\"
            )
        if cal == "TinyFace":
            clines.append("\\addlinespace")

    elines = []
    for cal in ("TinyFace", "XQLFW"):
        for name in ("DINOv2", "AdaFace"):
            r = pick(ablate_rows, cal=cal, family="embedding", embedding=name)
            elines.append(
                f"{cal} & {name} & {_fmt_r(r['r_hat'])} & {_fmt_c(r['set_size'])} \\\\"
            )
        if cal == "TinyFace":
            elines.append("\\addlinespace")

    empty_note = []
    for r in empty_rows:
        if abs(r["delta_r"]) > 1e-12 or abs(r["delta_c"]) > 1e-12:
            empty_note.append(
                f"{r['cal']} a={r['alpha']:.2f}: dR={r['delta_r']:.4g} dC={r['delta_c']:.4g}"
            )
    empty_txt = "; ".join(empty_note) if empty_note else "no change at any pair or alpha"

    tex = r"""% DS-CP baseline. Do not pack over frozen tables.
\begin{table}[t]
\centering\footnotesize
\caption{DS-CP (Lin et al., 2026) versus existing WCP-DC (MLP) on SCFace (Method B, $k=65$, $|G|=585$). DS-CP: XGBoost on frozen DINOv2, $\gamma=1$, in-sample weights, test-point mass $\max w$.}
\label{tab:dscp}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l c cc c cc @{}}
\toprule
&& \multicolumn{2}{c}{DS-CP} && \multicolumn{2}{c}{WCP-DC (MLP)} \\
\cmidrule{3-4}\cmidrule{6-7}
Pair & $\alpha$ & $\hat R$ & $|C|$ && $\hat R$ & $|C|$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table}

\begin{table}[t]
\centering\footnotesize
\caption{DS-CP weight quality versus unlabeled SCFace $t$ (XGBoost, DINOv2). TV and low-$t$ gap as in Table~\ref{tab:weight-quality}.}
\label{tab:dscp-weights}
\begin{tabular}{lrrrr}
\toprule
Pair & TV & Low-$t$ gap & ESS/$n$ & Weight range \\
\midrule
""" + "\n".join(dlines) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering\footnotesize
\caption{DS-CP ablations at $\alpha=0.10$ (Method B). Default: XGBoost, DINOv2, $\gamma=1$.}
\label{tab:dscp-ablation}
\begin{tabular}{llrr}
\toprule
Pair & Setting & $\hat R$ & $|C|$ \\
\midrule
\multicolumn{4}{l}{$\gamma$} \\
""" + "\n".join(glines) + r"""
\midrule
\multicolumn{4}{l}{Classifier (DINOv2, $\gamma=1$)} \\
""" + "\n".join(clines) + r"""
\midrule
\multicolumn{4}{l}{Embedding (XGBoost, $\gamma=1$)} \\
""" + "\n".join(elines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tables.tex").write_text(tex, encoding="utf-8")
    (OUT / "empty_set_note.txt").write_text(
        "Forcing non-empty sets (min_set_size=1, top-1 gallery image): " + empty_txt + "\n",
        encoding="utf-8",
    )
    log(f"Wrote {OUT / 'tables.tex'}")


def write_diff_note(empty_rows) -> None:
    changed = [
        r for r in empty_rows if abs(r["delta_r"]) > 1e-12 or abs(r["delta_c"]) > 1e-12
    ]
    empty = (
        "Forcing their top-1 fill (WACR min_set_size=1) left R-hat and |C| unchanged "
        "at every pair and alpha."
        if not changed
        else "Forcing their top-1 fill changed: "
        + "; ".join(
            f"{r['cal']} a={r['alpha']:.2f} dR={r['delta_r']:.4g} dC={r['delta_c']:.4g}"
            for r in changed
        )
        + "."
    )
    text = (
        "Our DS-CP follows src/cp_methods.py and src/main-mmlu.py at commit 60be9df, "
        "with these differences. The encoder is frozen DINOv2 ViT-B/14 (AdaFace in the "
        "sensitivity), not all-MiniLM-L6-v2, because this pipeline is image retrieval. "
        "Nonconformity is the Method B score (1-cos)/Phi, not LAC 1-p; one shared q-hat "
        "is still used for the whole SCFace batch. When the weighted cutoff exceeds the "
        "calibration mass their code sets q-hat=1, which is the full label set for LAC "
        "scores in [0,1]; we return +inf and the whole gallery of 585, the same vacuous "
        "event on a score that can exceed 1. Empty-set handling uses the WACR rule "
        "min_set_size=0 so DS-CP is comparable to the other baselines. "
        + empty
        + " We do not clip or rescale the odds to sum n or cap B=30; that cap is part of "
        "WCP-DC, not of their quantile. The test-point mass is gamma * max(w) with "
        "gamma=1, not WACR's B=30 atom. Classifier training is in-sample on all source "
        "plus all target embeddings, as in their code, with no cross-fitting."
    )
    (OUT / "diff_note.txt").write_text(text + "\n", encoding="utf-8")
    log(f"Wrote {OUT / 'diff_note.txt'}")


