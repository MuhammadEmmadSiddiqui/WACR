"""Uncertainty adapter Phi(x) for risk-aware retrieval."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def top1_fu_from_similarities(similarities: np.ndarray) -> np.ndarray:
    """Geometric uncertainty ``f_u(x) = clip(1 - max cosine sim, 0, 1)``."""
    return np.clip(1.0 - np.max(similarities, axis=1), 0.0, 1.0)


class UncertaintyAdapter:
    """Method-specific uncertainty transform to ``Phi(x)`` values.

    Methods:
      - ``mcd``: ``Phi(x) = clip((f_u(x) - fu_min) / (fu_max - fu_min), 0, 1) + beta``.
      - ``top1``: ``Phi(x) = clip(f_u(x), 0, 1) + beta`` where
                  ``f_u(x) = 1 - max_cosine_similarity``.
    """

    def __init__(
        self,
        method: str,
        beta: float,
        *,
        fu_min: Optional[float] = None,
        fu_max: Optional[float] = None,
    ) -> None:
        if method not in {"mcd", "top1"}:
            raise ValueError(f"Unknown uncertainty method: {method}")
        self.method = method
        self.beta = float(beta)
        self.fu_min = fu_min
        self.fu_max = fu_max

    @classmethod
    def fit(
        cls,
        *,
        method: str,
        beta: float,
        cal_fu: Optional[np.ndarray] = None,
        cal_sims: Optional[np.ndarray] = None,
    ) -> Tuple["UncertaintyAdapter", np.ndarray]:
        """Fit adapter and return ``(adapter, calibration_fu_used)``."""
        if method == "mcd":
            if cal_fu is None:
                raise ValueError("mcd adapter fit requires calibration fu values")
            fu_min = float(np.min(cal_fu))
            fu_max = float(np.max(cal_fu))
            if fu_max - fu_min <= 0:
                raise RuntimeError(
                    f"Degenerate calibration fu range (min=max={fu_min}); cannot fit Phi(x)."
                )
            adapter = cls(method=method, beta=beta, fu_min=fu_min, fu_max=fu_max)
            return adapter, np.asarray(cal_fu, dtype=np.float64)

        if cal_sims is None:
            raise ValueError("top1 adapter fit requires calibration similarity matrix")
        cal_fu_top1 = top1_fu_from_similarities(cal_sims)
        adapter = cls(method=method, beta=beta)
        return adapter, np.asarray(cal_fu_top1, dtype=np.float64)

    def transform(self, fu: np.ndarray) -> np.ndarray:
        fu = np.asarray(fu, dtype=np.float64).reshape(-1)
        if self.method == "mcd":
            if self.fu_min is None or self.fu_max is None:
                raise RuntimeError("mcd adapter used before fitting fu_min/fu_max")
            scaled = (fu - self.fu_min) / (self.fu_max - self.fu_min + 1e-12)
            return np.clip(scaled, 0.0, 1.0) + self.beta
        return np.clip(fu, 0.0, 1.0) + self.beta
