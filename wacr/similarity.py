"""Cosine similarity helpers for embedding matrices."""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalized cosine similarity between query rows A and gallery B."""
    return l2_normalize(a) @ l2_normalize(b).T
