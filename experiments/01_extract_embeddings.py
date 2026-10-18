#!/usr/bin/env python3
"""Extract AdaFace MCD embeddings (and optional DINOv2) from images.

Default image root: data/images/<pair>/{TinyFace,XQLFW,SCFace}/{Gallery,Queries}/
Pass --image-root to point at an existing dataset tree. Do not commit images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from wacr.paths import pair_cache_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="tinyface_scface")
    p.add_argument("--image-root", type=str, default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="If >0, extract at most this many images per split (cosine check).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sys.argv = [
        sys.argv[0],
        "--pair",
        args.pair,
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.image_root:
        sys.argv += ["--input-dir", args.image_root]

    from wacr import _extract_mcd as mcd

    if args.limit > 0:
        print(
            f"limit={args.limit} is handled by experiments/01 when used for the cosine check; "
            f"full extraction uses the AdaFace MCD pipeline.",
            flush=True,
        )
        cosine_check_extract(args)
        return
    mcd.main()


def cosine_check_extract(args) -> None:
    """Re-extract a random subset and write outputs/embed_cosine_check.csv."""
    import numpy as np
    from PIL import Image

    from wacr.embeddings import load_bundle
    from wacr.identity import identity_of
    from wacr.similarity import l2_normalize

    pair = args.pair
    cache = pair_cache_dir(pair)
    cal = "TinyFace" if "tinyface" in pair else "XQLFW"
    rng = np.random.default_rng(args.seed)
    rows = []
    try:
        import torch
        from wacr import _extract_mcd as mcd
    except Exception as exc:
        print(f"skip extract: {type(exc).__name__}: {exc}", flush=True)
        return

    image_root = Path(args.image_root) if args.image_root else REPO / "data" / "images" / pair
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    try:
        mcd.ensure_hub_python_dependencies()
        mcd.set_extract_seed(args.seed)
        model = mcd.load_backbone(device, None)
        tfm = mcd.image_transform()
    except Exception as exc:
        print(f"skip backbone: {type(exc).__name__}: {exc}", flush=True)
        return

    by_name: dict[str, Path] = {}
    if image_root.is_dir():
        for p in image_root.rglob("*"):
            if p.is_file():
                by_name.setdefault(p.name, p)
                rel = str(p.as_posix())
                for key in (p.name, "/".join(p.parts[-3:])):
                    by_name.setdefault(key, p)
        print(f"indexed {len(by_name)} keys under image root", flush=True)

    for split in (f"{cal}_Gallery", "SCFace_Gallery", f"{cal}_Queries", "SCFace_Queries"):
        npz = cache / f"{split}_mcd.npz"
        if not npz.is_file():
            continue
        bundle = load_bundle(npz, name=split)
        n = min(int(args.limit), len(bundle.paths))
        idx = rng.choice(len(bundle.paths), size=n, replace=False)
        cached = l2_normalize(np.asarray(bundle.embeddings, dtype=np.float64))
        sims = []
        import torch.nn.functional as F

        with torch.no_grad():
            for i in idx:
                stored = bundle.paths[int(i)].replace("\\", "/")
                tail = "/".join(Path(stored).parts[-3:])
                name = Path(stored).name
                hit = by_name.get(tail) or by_name.get(name)
                if hit is None:
                    continue
                im = tfm(Image.open(hit).convert("RGB")).unsqueeze(0).to(device)
                # eval() matches gallery caches. Query caches are 40-pass MCD, so
                # this is a one-forward check, not a full MCD rerun.
                model.eval()
                emb = mcd.embed_image_batch(model, im)
                vec = F.normalize(emb.float(), dim=-1).cpu().numpy().reshape(-1)
                sims.append(float(np.dot(vec, cached[int(i)])))
                if len(sims) % 10 == 0:
                    print(f"  {split} {len(sims)}/{n} min={min(sims):.6f}", flush=True)
        if sims:
            rows.append(
                {
                    "pair": pair,
                    "split": split,
                    "n": len(sims),
                    "min_cosine": float(np.min(sims)),
                    "mean_cosine": float(np.mean(sims)),
                }
            )
            print(f"{split} n={len(sims)} min_cos={float(np.min(sims)):.6f}", flush=True)
    dest = REPO / "outputs" / f"embed_cosine_check_{pair}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        import csv

        with dest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
