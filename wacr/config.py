"""Load configs/default.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from wacr.paths import REPO_ROOT


def load_config(path: Path | None = None) -> dict:
    p = path or (REPO_ROOT / "configs" / "default.yaml")
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
