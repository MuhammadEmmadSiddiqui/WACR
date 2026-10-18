#!/usr/bin/env python3
"""Download AdaFace (official Hub snapshot) and DINOv2 (torch.hub). Do not commit weights."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "checkpoints"
ADA_ID = "minchul/cvlface_adaface_vit_base_webface4m"
DINO_REPO = "facebookresearch/dinov2"
DINO_MODEL = "dinov2_vitb14"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adaface-only", action="store_true")
    p.add_argument("--dinov2-only", action="store_true")
    args = p.parse_args()
    CKPT.mkdir(parents=True, exist_ok=True)
    if not args.dinov2_only:
        from huggingface_hub import snapshot_download

        dest = snapshot_download(ADA_ID, cache_dir=str(CKPT / "hf"))
        print(f"AdaFace snapshot: {dest}")
    if not args.adaface_only:
        import torch

        torch.hub.load(DINO_REPO, DINO_MODEL)
        print("DINOv2 via torch.hub OK")


if __name__ == "__main__":
    main()
