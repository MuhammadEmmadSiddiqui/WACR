"""NPZ embedding cache loaders (``cache/<combination>/`` layout)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from wacr.identity import identity_of
from wacr.weights import decode_paths

EMBEDDING_KEYS = ("embeddings", "features", "fe")


@dataclass
class EmbeddingBundle:
    """In-memory view of one ``*_mcd.npz`` cache."""

    name: str
    embeddings: np.ndarray            # (N, D) float32
    paths: List[str]
    fu: Optional[np.ndarray] = None   # (N,) float64 when available

    @property
    def identities(self) -> List[str]:
        return [identity_of(p) for p in self.paths]


def _load_embedding_array(data: np.lib.npyio.NpzFile, npz_path: Path) -> np.ndarray:
    for key in EMBEDDING_KEYS:
        if key in data.files:
            arr = np.asarray(data[key], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"{npz_path}: key '{key}' has shape {arr.shape}; expected 2D"
                )
            return arr
    raise KeyError(
        f"{npz_path}: none of {EMBEDDING_KEYS} found. Available keys: {list(data.files)}"
    )


def load_bundle(npz_path: Path, *, name: str) -> EmbeddingBundle:
    """Load query or gallery embeddings, paths, and optional ``fu``."""
    data = np.load(npz_path, allow_pickle=True)
    if "paths" not in data.files:
        raise KeyError(f"{npz_path}: missing 'paths'; have {list(data.files)}")

    embeddings = _load_embedding_array(data, npz_path)
    paths = [str(p) for p in np.asarray(data["paths"], dtype=object).reshape(-1)]
    if embeddings.shape[0] != len(paths):
        raise ValueError(
            f"{npz_path}: {embeddings.shape[0]} embedding rows vs {len(paths)} paths"
        )

    fu: Optional[np.ndarray] = None
    if "fu" in data.files:
        fu = np.asarray(data["fu"], dtype=np.float64).reshape(-1)
        if fu.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"{npz_path}: fu length {fu.shape[0]} != embedding rows {embeddings.shape[0]}"
            )

    return EmbeddingBundle(name=name, embeddings=embeddings, paths=paths, fu=fu)


def load_queries_mcd(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load query embeddings ``fe``, uncertainty ``fu``, and ``paths`` as float64."""
    z = np.load(path, allow_pickle=True)
    for k in ("fe", "fu", "paths"):
        if k not in z.files:
            raise KeyError(f"{path}: missing {k!r}; have {list(z.files)}")
    fe = np.asarray(z["fe"], dtype=np.float64)
    fu = np.asarray(z["fu"], dtype=np.float64).reshape(-1)
    paths = decode_paths(z["paths"])
    n = fe.shape[0]
    if fu.shape[0] != n or len(paths) != n:
        raise RuntimeError(f"{path}: fe/fu/paths length mismatch (n={n})")
    return fe, fu, paths


def load_gallery_mcd(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load gallery embeddings ``fe`` and ``paths`` (no per-item uncertainty)."""
    z = np.load(path, allow_pickle=True)
    for k in ("fe", "paths"):
        if k not in z.files:
            raise KeyError(f"{path}: missing {k!r}; have {list(z.files)}")
    fe = np.asarray(z["fe"], dtype=np.float64)
    paths = decode_paths(z["paths"])
    if fe.shape[0] != len(paths):
        raise RuntimeError(f"{path}: len(fe) vs len(paths) mismatch")
    return fe, paths


def load_embeddings_and_paths(
    npz_path: Path,
    *,
    dtype: type = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a 2D (N, D) embedding matrix and aligned ``paths`` for weight scripts.

    Accepts any of ``EMBEDDING_KEYS`` (``fe`` in MCD caches, ``embeddings`` in
    legacy archives). Returns ``paths`` as a 1D object array for direct NPZ export.
    """
    data = np.load(npz_path, allow_pickle=True)

    embeddings = None
    for key in EMBEDDING_KEYS:
        if key in data.files:
            arr = np.asarray(data[key], dtype=dtype)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim != 2:
                raise ValueError(
                    f"{npz_path}: key '{key}' has shape {arr.shape}; expected (N, D)"
                )
            embeddings = arr
            break

    if embeddings is None:
        raise KeyError(
            f"{npz_path}: none of {EMBEDDING_KEYS} found. "
            f"Available keys: {list(data.files)}"
        )

    if "paths" not in data.files:
        raise KeyError(
            f"{npz_path}: missing 'paths'. Available keys: {list(data.files)}"
        )

    paths = np.asarray(data["paths"], dtype=object).reshape(-1)
    if embeddings.shape[0] != len(paths):
        raise ValueError(
            f"{npz_path}: {embeddings.shape[0]} embedding rows vs {len(paths)} paths"
        )

    return embeddings, paths


def load_embeddings(npz_path: Path, *, dtype: type = np.float64) -> np.ndarray:
    """Embedding matrix only (no paths); see ``load_embeddings_and_paths``."""
    embeddings, _ = load_embeddings_and_paths(npz_path, dtype=dtype)
    return embeddings
