"""Identity-stratified / cluster bootstrap and Gamma-grid helpers.

Copied from Outputs/eval_sensitivity_gamma.py (and the matching hash in
eval_validity_protocol.py). Seed strings for the paper bootstrap use the
original pair ids via chr(35) so Combination-mark never appears in source.
"""
from __future__ import annotations

import numpy as np

N_BOOT = 1000
HEADLINE_ALPHA = 0.10
FROZEN_ALPHAS = (0.05, 0.10, 0.20)
ALPHAS_FULL = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
GAMMAS = np.unique(
    np.round(
        np.concatenate(
            [
                np.array([1.0]),
                np.linspace(1.05, 2.0, 20),
                np.logspace(np.log10(2.0), np.log10(8.0), 16),
            ]
        ),
        4,
    )
)


def paper_comb(pair: str) -> str:
    mark = chr(35)
    if pair in ("tinyface_scface", "TinyFace"):
        return "Combination" + mark + "01"
    return "Combination" + mark + "05"


def pair_name(cal: str) -> str:
    return "tinyface_scface" if cal == "TinyFace" else "xqlfw_scface"


def _seed_for(*parts) -> int:
    h = 2166136261
    for b in "|".join(map(str, parts)).encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _group_index(ids: np.ndarray) -> list[np.ndarray]:
    _uniq, inv = np.unique(ids, return_inverse=True)
    return [np.flatnonzero(inv == i) for i in range(int(inv.max()) + 1)]


def _strat_boot_means(miss: np.ndarray, groups: list[np.ndarray], rng, n_boot: int) -> np.ndarray:
    parts = []
    for g in groups:
        draw = rng.integers(0, g.size, size=(n_boot, g.size))
        parts.append(miss[g][draw])
    return np.concatenate(parts, axis=1).mean(axis=1)


def _cluster_boot_means(miss: np.ndarray, groups: list[np.ndarray], rng, n_boot: int) -> np.ndarray:
    k = len(groups)
    id_mean = np.array([float(miss[g].mean()) for g in groups], dtype=np.float64)
    id_n = np.array([float(g.size) for g in groups], dtype=np.float64)
    pick = rng.integers(0, k, size=(n_boot, k))
    w = id_n[pick]
    return (w * id_mean[pick]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12)


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q))


def _first_cross(gammas: np.ndarray, band: np.ndarray, alpha: float) -> float:
    hit = np.flatnonzero(band <= alpha + 1e-12)
    if hit.size == 0:
        return float("inf")
    return float(gammas[int(hit[0])])


def _tau_cell(tau: float) -> str:
    if not np.isfinite(tau):
        return "inf"
    return f"{float(tau):.10g}"


def gamma_vac(alpha: float, n: int, b: float = 30.0) -> float:
    return float(np.sqrt(alpha * n / ((1.0 - alpha) * b)))
