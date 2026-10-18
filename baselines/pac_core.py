"""PAC-CS (Park et al. 2022 PS-W) and Huang KMM QP as copied from eval_pac_published.py."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel

from baselines.ds_cp import l2_rows
from wacr.eval_b import _eval_b, empty_pct

DELTA = 0.1
SEED = 0
K_BINS = 10
E_SMOOTH = 0.001
KMM_CAP = 10_000.0
KMM_K = 10
MLP_HIDDEN = 500
MLP_DROPOUT = 0.5
MLP_EPOCHS = 80
MLP_BATCH = 256
MLP_LR = 1e-3
MLP_PATIENCE = 12
IW_EPS = 1e-6
N_GALLERY = 585


def clopper_pearson_ucb(k: int, n: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - delta, k + 1, n - k))


def cp_interval(k: int, n: int, delta0: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    k = int(k)
    n = int(n)
    delta0 = float(min(max(delta0, 1e-15), 1.0 - 1e-15))
    if k <= 0:
        lo = 0.0
    else:
        lo = float(beta_dist.ppf(delta0, k, n - k + 1))
        if not np.isfinite(lo):
            lo = 0.0
    if k >= n:
        hi = 1.0
    else:
        hi = float(beta_dist.ppf(1.0 - delta0, k + 1, n - k))
        if not np.isfinite(hi):
            hi = 1.0
    return float(lo), float(hi)


def ess_of(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return 0.0
    return float((w.sum() ** 2) / max(float((w ** 2).sum()), 1e-12))


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


class DomainMLP:
    """Park MidFNN: Linear-ReLU-Dropout x2, then Linear to 2 logits."""

    def __init__(self, d_in: int, seed: int = 0):
        torch, nn, _F = _torch()
        self.torch = torch
        self.nn = nn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = torch.Generator()
        g.manual_seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        class Net(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, MLP_HIDDEN),
                    nn.ReLU(),
                    nn.Dropout(MLP_DROPOUT),
                    nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
                    nn.ReLU(),
                    nn.Dropout(MLP_DROPOUT),
                    nn.Linear(MLP_HIDDEN, 2),
                )

            def forward(self, x):
                return self.net(x)

        self.net = Net(int(d_in)).to(self.device)
        self.seed = int(seed)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = MLP_EPOCHS,
    ) -> dict:
        torch = self.torch
        x_t = torch.tensor(np.asarray(x, dtype=np.float32), device=self.device)
        y_t = torch.tensor(np.asarray(y, dtype=np.int64), device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=MLP_LR)
        n = int(x_t.shape[0])
        best_auc = -1.0
        best_state = None
        best_ep = 0
        stall = 0
        rng = np.random.default_rng(self.seed + 17)
        for ep in range(int(epochs)):
            self.net.train()
            perm = rng.permutation(n)
            total = 0.0
            nbat = 0
            for s in range(0, n, MLP_BATCH):
                idx = perm[s : s + MLP_BATCH]
                logits = self.net(x_t[idx])
                loss = torch.nn.functional.cross_entropy(logits, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                nbat += 1
            auc = float("nan")
            if x_val is not None and y_val is not None and y_val.size and np.unique(y_val).size > 1:
                p = self.predict_proba(x_val)
                auc = float(roc_auc_score(y_val, p[:, 1]))
                if auc > best_auc + 1e-5:
                    best_auc = auc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    best_ep = ep + 1
                    stall = 0
                else:
                    stall += 1
                    if stall >= MLP_PATIENCE:
                        break
        if best_state is not None:
            self.net.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return {
            "epochs_ran": ep + 1,
            "best_epoch": best_ep,
            "val_auc": best_auc if best_auc >= 0 else float("nan"),
            "mean_loss": total / max(nbat, 1),
            "device": str(self.device),
        }

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        x_t = torch.tensor(np.asarray(x, dtype=np.float32), device=self.device)
        out = []
        with torch.no_grad():
            for s in range(0, x_t.shape[0], 2048):
                logits = self.net(x_t[s : s + 2048])
                p = torch.softmax(logits, dim=1)
                out.append(p.cpu().numpy())
        return np.concatenate(out, axis=0)


def identity_holdout(ids: np.ndarray, frac: float, seed: int) -> np.ndarray:
    ids = np.asarray(ids)
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    n_te = max(1, int(round(frac * float(uniq.size))))
    n_te = min(n_te, max(1, int(uniq.size) - 1)) if uniq.size > 1 else 1
    te = set(rng.choice(uniq, size=n_te, replace=False).tolist())
    return np.array([i in te for i in ids], dtype=bool)


def heuristic_iw(p_source: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    g = np.clip(np.asarray(p_source, dtype=np.float64).reshape(-1), IW_EPS, 1.0 - IW_EPS)
    w = 1.0 / g - 1.0
    w = w + rng.random(w.shape[0]) * IW_EPS
    return w


def binedges_equalmass(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    n = int(x.size)
    if n == 0:
        return np.array([0.0, np.inf], dtype=np.float64)
    edges = np.interp(np.linspace(0.0, n, int(n_bins) + 1), np.arange(n, dtype=np.float64), x)
    edges[0] = 0.0
    edges[-1] = np.inf
    for i in range(1, edges.size - 1):
        if not np.isfinite(edges[i]) or edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def assign_bins(w: np.ndarray, edges: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    idx = np.zeros(w.size, dtype=np.int64)
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if j == edges.size - 2:
            m = (w >= lo) & (w <= hi)
        else:
            m = (w >= lo) & (w < hi)
        idx[m] = j
    return idx


def theorem5_bounds(
    w_src: np.ndarray,
    w_tgt: np.ndarray,
    *,
    k_bins: int = K_BINS,
    delta_w: float = DELTA / 2.0,
    e: float = E_SMOOTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    edges = binedges_equalmass(w_src, k_bins)
    src_bin = assign_bins(w_src, edges)
    tgt_bin = assign_bins(w_tgt, edges)
    m = int(w_src.size)
    n = int(w_tgt.size)
    delta0 = float(delta_w) / (2.0 * float(k_bins))
    lo = np.zeros(k_bins, dtype=np.float64)
    hi = np.ones(k_bins, dtype=np.float64)
    n_src = np.zeros(k_bins, dtype=np.int64)
    n_tgt = np.zeros(k_bins, dtype=np.int64)
    for j in range(k_bins):
        ns = int(np.sum(src_bin == j))
        nt = int(np.sum(tgt_bin == j))
        n_src[j] = ns
        n_tgt[j] = nt
        p_lo, p_hi = cp_interval(ns, m, delta0)
        q_lo, q_hi = cp_interval(nt, n, delta0)
        lo[j] = max(0.0, q_lo - e) / (p_hi + e + 1e-16)
        hi[j] = (q_hi + e) / (max(0.0, p_lo - e) + 1e-16)
        if not np.isfinite(lo[j]):
            lo[j] = 0.0
        if not np.isfinite(hi[j]):
            hi[j] = 1e12
        hi[j] = min(float(hi[j]), 1e12)
    w_lo = lo[src_bin]
    w_hi = hi[src_bin]
    meta = {
        "k_bins": int(k_bins),
        "delta0": delta0,
        "b": float(np.max(hi)),
        "n_src_bins": ",".join(str(int(x)) for x in n_src),
        "n_tgt_bins": ",".join(str(int(x)) for x in n_tgt),
        "w_lo_min": float(np.min(lo)),
        "w_hi_max": float(np.max(hi)),
        "edges": ",".join(f"{float(x):.6g}" if np.isfinite(x) else "inf" for x in edges),
    }
    return w_lo, w_hi, edges, meta


def ps_w_tau(
    scores: np.ndarray,
    w_lo: np.ndarray,
    w_hi: np.ndarray,
    u: np.ndarray,
    alpha: float,
    delta_c: float,
) -> tuple[float, int, int, float, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    w_lo = np.asarray(w_lo, dtype=np.float64).reshape(-1)
    w_hi = np.asarray(w_hi, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    b = float(np.nanmax(w_hi))
    if not np.isfinite(b) or b <= 0.0:
        return float("inf"), 0, 0, 1.0, b
    b = max(b, 1e-12)
    n = int(scores.size)
    acc = u <= (w_hi / b)
    n_acc = int(acc.sum())
    k = n_acc
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    i = 0
    while i < n:
        tau = float(s[i])
        j = i
        while j < n and s[j] == tau:
            t = int(order[j])
            if acc[t]:
                n_acc -= 1
                k -= 1
            now = bool(u[t] <= (w_lo[t] / b))
            acc[t] = now
            if now:
                n_acc += 1
            j += 1
        ucb = clopper_pearson_ucb(k, n_acc, delta_c)
        if ucb <= float(alpha) + 1e-12:
            return tau, n_acc, k, float(ucb), b
        i = j
    ucb = clopper_pearson_ucb(k, n_acc, delta_c)
    return float("inf"), n_acc, k, float(ucb), b


def train_domain_classifier(
    x_src: np.ndarray,
    x_tgt: np.ndarray,
    src_ids: np.ndarray,
    tgt_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x_src = l2_rows(x_src)
    x_tgt = l2_rows(x_tgt)
    src_hold = identity_holdout(src_ids, 0.2, seed + 3)
    tgt_hold = identity_holdout(tgt_ids, 0.2, seed + 5)
    x_tr = np.vstack([x_src[~src_hold], x_tgt[~tgt_hold]])
    y_tr = np.concatenate(
        [
            np.ones(int((~src_hold).sum()), dtype=np.int64),
            np.zeros(int((~tgt_hold).sum()), dtype=np.int64),
        ]
    )
    x_va = np.vstack([x_src[src_hold], x_tgt[tgt_hold]])
    y_va = np.concatenate(
        [
            np.ones(int(src_hold.sum()), dtype=np.int64),
            np.zeros(int(tgt_hold.sum()), dtype=np.int64),
        ]
    )
    d = int(x_src.shape[1])
    clf_va = DomainMLP(d, seed=seed)
    va = clf_va.fit(x_tr, y_tr, x_va, y_va)
    auc = float(va["val_auc"])
    clf = DomainMLP(d, seed=seed + 1)
    full = clf.fit(
        np.vstack([x_src, x_tgt]),
        np.concatenate(
            [np.ones(x_src.shape[0], dtype=np.int64), np.zeros(x_tgt.shape[0], dtype=np.int64)]
        ),
        x_va,
        y_va,
    )
    p_src = clf.predict_proba(x_src)[:, 1]
    p_tgt = clf.predict_proba(x_tgt)[:, 1]
    return p_src, p_tgt, {
        "auc_holdout": auc,
        "epochs_full": full["epochs_ran"],
        "device": full["device"],
        "best_epoch": full["best_epoch"],
    }


def stats_from_tau(wacr, job, tau: float) -> dict:
    r, c, miss, sizes = _eval_b(wacr, job, tau)
    n_eval = int(job["n_eval"])
    n_empty = int(np.sum(np.asarray(sizes) <= 1e-12))
    n_full = int(np.sum(np.asarray(sizes) + 1e-9 >= N_GALLERY))
    return {
        "r_hat": float(r),
        "set_size": float(c),
        "tau": float(tau),
        "tau_inf": int(not np.isfinite(tau)),
        "empty_pct": 100.0 * n_empty / max(n_eval, 1),
        "n_empty": n_empty,
        "n_full": n_full,
        "n_eval": n_eval,
        "miss": miss,
        "sizes": sizes,
    }


def kmm_weights_qp(
    x_src: np.ndarray,
    x_tgt: np.ndarray,
    *,
    cap: float = KMM_CAP,
    seed: int = 0,
    median_pairwise_sigma=None,
) -> tuple[np.ndarray, dict]:
    import cvxpy as cp

    x_src = np.asarray(x_src, dtype=np.float64)
    x_tgt = np.asarray(x_tgt, dtype=np.float64)
    n = int(x_src.shape[0])
    m = int(x_tgt.shape[0])
    pooled = np.vstack([x_src, x_tgt])
    sigma = median_pairwise_sigma(pooled, seed=seed)
    gamma = 1.0 / max(sigma * sigma, 1e-12)
    eps = 1.0 - 1.0 / math.sqrt(float(n))
    k_mat = rbf_kernel(x_src, x_src, gamma=gamma)
    k_mat = 0.5 * (k_mat + k_mat.T) + 1e-8 * np.eye(n)
    k_st = rbf_kernel(x_src, x_tgt, gamma=gamma)
    kappa = (n / float(m)) * np.sum(k_st, axis=1)
    w = cp.Variable(n)
    obj = 0.5 * cp.quad_form(w, cp.psd_wrap(k_mat)) - kappa @ w
    cons = [
        w >= 0,
        w <= float(cap),
        cp.sum(w) >= n * (1.0 - eps),
        cp.sum(w) <= n * (1.0 + eps),
    ]
    prob = cp.Problem(cp.Minimize(obj), cons)
    status = "unsolved"
    value = None
    t0 = time.perf_counter()
    attempts = [
        ("OSQP", dict(solver=cp.OSQP, verbose=False, max_iter=20000, eps_abs=1e-4, eps_rel=1e-4, polish=True)),
        ("CLARABEL", dict(solver=cp.CLARABEL, verbose=False, max_iter=1500)),
        ("SCS", dict(solver=cp.SCS, verbose=False, max_iters=5000)),
    ]
    for name, kw in attempts:
        try:
            prob.solve(**kw)
            st = str(prob.status)
            if w.value is not None and np.all(np.isfinite(w.value)):
                value = np.asarray(w.value, dtype=np.float64).reshape(-1)
                status = f"{name}:{st}"
                if "optimal" in st.lower() or st in {"optimal", "optimal_inaccurate"}:
                    break
            else:
                status = f"{name}:{st}:no_value"
        except Exception as exc:
            status = f"{name}:fail:{type(exc).__name__}"
    elapsed = time.perf_counter() - t0
    if value is None:
        value = np.full(n, np.nan, dtype=np.float64)
        ok = False
    else:
        value = np.clip(value, 0.0, float(cap))
        ok = bool(np.all(np.isfinite(value)))
    meta = {
        "sigma": float(sigma),
        "gamma": float(gamma),
        "epsilon": float(eps),
        "status": status,
        "ok": int(ok),
        "seconds": float(elapsed),
        "mean_w": float(np.nanmean(value)) if value.size else float("nan"),
        "max_w": float(np.nanmax(value)) if value.size else float("nan"),
        "min_w": float(np.nanmin(value)) if value.size else float("nan"),
        "ess": ess_of(np.nan_to_num(value, nan=0.0)),
        "n": n,
        "m": m,
        "cap": float(cap),
    }
    return value, meta
