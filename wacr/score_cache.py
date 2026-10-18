"""Score-level cache: t, Method B scores, baseline weights. No images, no embeddings."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from wacr.paths import REPO_ROOT, pair_cache_dir
from wacr.protocol import JOBS
from wacr.weights import load_importance_weights

SHARE_DIR = REPO_ROOT / "cache_share"
ENV_FLAG = "WACR_FROM_SCORES"

_ENABLED = False
_GUARD = False


def using_scores() -> bool:
    if _ENABLED:
        return True
    v = os.environ.get(ENV_FLAG, "").strip().lower()
    return v in {"1", "true", "yes"}


def enable() -> None:
    global _ENABLED, _GUARD
    _ENABLED = True
    os.environ[ENV_FLAG] = "1"
    if _GUARD:
        return
    import sys

    import wacr.embeddings as em

    def _block(npz_path, *args, **kwargs):
        raise RuntimeError(f"from-scores mode: embedding cache is blocked ({npz_path})")

    original = em.load_bundle
    em.load_bundle = _block  # type: ignore[assignment]
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, "load_bundle", None) is original:
                setattr(mod, "load_bundle", _block)
        except Exception:
            continue
    _GUARD = True


def pack_path(comb: str) -> Path:
    return SHARE_DIR / f"{comb}.npz"


def _obj(seq) -> np.ndarray:
    return np.asarray(list(seq), dtype=object)


def _maybe_load_w(cache: Path, cal: str, tag: str, paths: list[str]):
    p = cache / f"{cal}_Queries_{tag}.npz"
    if not p.is_file():
        return None
    return load_importance_weights(p, paths)


def job_from_pack(wacr, comb: str, cal: str, tst: str) -> dict:
    path = pack_path(comb)
    if not path.is_file():
        raise FileNotFoundError(f"score cache missing: {path}")
    z = np.load(path, allow_pickle=True)

    def arr(key, dtype=None):
        a = z[key]
        if dtype is None:
            return a
        return np.asarray(a, dtype=dtype)

    def opt(key, dtype=np.float64):
        if key not in z.files:
            return None
        a = np.asarray(z[key], dtype=dtype)
        if a.size == 1 and not np.isfinite(a.reshape(-1)[0]):
            return None
        return a

    calib_paths = [str(p) for p in z["calib_paths"]]
    test_paths = [str(p) for p in z["test_paths"]]
    rec = {
        "comb": comb,
        "cal": cal,
        "tst": tst,
        "cache": pair_cache_dir(comb),
        "calib_paths": calib_paths,
        "calib_ids": arr("calib_ids", object),
        "cg_ids": arr("cg_ids", object),
        "s_base": arr("s_base", np.float64),
        "s_g": arr("s_g", np.float64),
        "t_cal": arr("t_cal", np.float64),
        "phi_matched": arr("phi_matched", np.float64),
        "phi_cal": arr("phi_cal", np.float64),
        "test_ids": arr("test_ids", object),
        "test_qids": arr("test_qids", object),
        "tg_ids": arr("tg_ids", object),
        "test_paths": test_paths,
        # float32 matches load_job cosine matmul; float64 1-sim ties differ
        "test_sims": arr("test_sims", np.float32),
        "true_test": arr("true_test", np.int64),
        "phi_test": arr("phi_test", np.float64),
        "t_te": arr("t_te", np.float64),
        "s_b_test": arr("s_b_test", np.float64),
        "cos_true": arr("cos_true", np.float64),
        "top1_full": arr("top1_full", np.float64),
        "n_eval": int(z["n_eval"]),
        "n_calib": int(z["n_calib"]),
        "n_ids_test": int(z["n_ids_test"]),
        "test_valid": arr("test_valid", bool),
        "calib_sims": arr("calib_sims", np.float32),
        "w_kmm": opt("w_kmm"),
        "w_kliep": opt("w_kliep"),
        "w_dc": opt("w_dc"),
        "w_paccs": opt("w_paccs"),
        "w_paccs_lo": opt("w_paccs_lo"),
        "w_paccs_hi": opt("w_paccs_hi"),
        "u_paccs": opt("u_paccs"),
        "w_ltt": opt("w_ltt"),
        "w_dscp": opt("w_dscp"),
        "w_wrcp_cal": opt("w_wrcp_cal"),
        "w_wrcp_te": opt("w_wrcp_te"),
        "w_wrcp_fold": opt("w_wrcp_fold", np.int32),
        "w_wrcp_keep": opt("w_wrcp_keep", np.int64),
        "from_scores": True,
    }
    rec["n_eval"] = int(rec["test_valid"].sum())
    rec["n_calib"] = int(rec["s_g"].shape[0])
    return rec


def dump_job(job: dict) -> Path:
    attach_file_weights(job)
    SHARE_DIR.mkdir(parents=True, exist_ok=True)
    path = pack_path(job["comb"])
    phi = np.asarray(job["phi_test"], dtype=np.float64).reshape(-1)
    sims = np.asarray(job["test_sims"])
    s_b_test = (1.0 - sims.astype(np.float64)) / np.maximum(phi[:, None], 1e-12)
    phi_m = np.asarray(job["phi_matched"], dtype=np.float64).reshape(-1)
    calib_sims = np.asarray(job["calib_sims"])
    s_b_matched = (1.0 - calib_sims) / np.maximum(phi_m[:, None], 1e-12)
    payload = dict(
        calib_paths=_obj(job["calib_paths"]),
        calib_ids=_obj(job["calib_ids"]),
        cg_ids=_obj(job["cg_ids"]),
        s_base=np.asarray(job["s_base"], dtype=np.float64),
        s_g=np.asarray(job["s_g"], dtype=np.float64),
        t_cal=np.asarray(job["t_cal"], dtype=np.float64),
        phi_matched=np.asarray(job["phi_matched"], dtype=np.float64),
        phi_cal=np.asarray(job["phi_cal"], dtype=np.float64),
        test_ids=_obj(job["test_ids"]),
        test_qids=_obj(job["test_qids"]),
        tg_ids=_obj(job["tg_ids"]),
        test_paths=_obj(job["test_paths"]),
        test_sims=sims,
        s_b_matched=np.asarray(s_b_matched, dtype=np.float64),
        s_b_test=np.asarray(s_b_test, dtype=np.float64),
        true_test=np.asarray(job["true_test"], dtype=np.int64),
        phi_test=phi,
        t_te=np.asarray(job["t_te"], dtype=np.float64),
        cos_true=np.asarray(job["cos_true"], dtype=np.float64),
        top1_full=np.asarray(job["top1_full"], dtype=np.float64),
        n_eval=np.int64(job["n_eval"]),
        n_calib=np.int64(job["n_calib"]),
        n_ids_test=np.int64(job["n_ids_test"]),
        test_valid=np.asarray(job["test_valid"], dtype=bool),
        calib_sims=calib_sims,
    )
    for key in (
        "w_kmm",
        "w_kliep",
        "w_dc",
        "w_paccs",
        "w_paccs_lo",
        "w_paccs_hi",
        "u_paccs",
        "w_ltt",
        "w_dscp",
        "w_wrcp_cal",
        "w_wrcp_te",
        "w_wrcp_fold",
        "w_wrcp_keep",
    ):
        val = job.get(key)
        if val is not None:
            payload[key] = np.asarray(val)
    np.savez_compressed(path, **payload)
    print(f"Wrote {path}", flush=True)
    return path


def load_jobs_from_scores(wacr) -> list[dict]:
    return [job_from_pack(wacr, *spec) for spec in JOBS]


def attach_file_weights(job: dict) -> None:
    cache, cal, paths = job["cache"], job["cal"], job["calib_paths"]
    if job.get("w_kmm") is None:
        job["w_kmm"] = _maybe_load_w(cache, cal, "kmm_median", paths)
    if job.get("w_kliep") is None:
        job["w_kliep"] = _maybe_load_w(cache, cal, "kliep", paths)
    if job.get("w_dc") is None:
        job["w_dc"] = _maybe_load_w(cache, cal, "wx", paths)
