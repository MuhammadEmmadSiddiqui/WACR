# Data

This repository does not distribute images. Embedding caches used by the paper live under `cache/<pair>/` (gitignored). TinyFace manifests list original QMUL relative paths and identity IDs (no SHA-256). XQLFW and SCFace manifests list file names, identity IDs, and SHA-256 of the local image bytes.

## Pairs

- `tinyface_scface`: TinyFace calibration, SCFace target
- `xqlfw_scface`: XQLFW calibration, SCFace target

Expected image layout if you extract embeddings from scratch (`run_all.py --from-images`):

```
data/images/tinyface_scface/TinyFace/{Gallery,Queries}/
data/images/tinyface_scface/SCFace/{Gallery,Queries}/
data/images/xqlfw_scface/XQLFW/{Gallery,Queries}/
data/images/xqlfw_scface/SCFace/{Gallery,Queries}/
```

SCFace retrieval queries are identities 066-130 (nine views). Gallery size is 585. Identities 001-065 are used only for gallery construction / classifier-pool notes; they are not retrieval queries.

## Downloads (public datasets)

- TinyFace: original QMUL TinyFace release (unconstrained face identification). Follow the dataset licence. The AdaFace caches were extracted from that original release; tracked TinyFace manifests use the relative paths stored in those caches (`TinyFace/Gallery/...`, `TinyFace/Queries/...`).
- XQLFW: https://martlgap.github.io/xqlfw/ (cross-quality LFW). Follow the dataset licence.
- SCFace: request from the SCFace distributors; academic licence.

Place files so that stems match `data/manifests/*.csv`.

## How the embedding caches were made (no new extraction)

All AdaFace caches were written by `Inference/extract_mcd_features.py` (this repo: `wacr/_extract_mcd.py`).

- **Model:** Hub id `minchul/cvlface_adaface_vit_base_webface4m` (AdaFace ViT-Base, WebFace4M).
- **Preprocessing:** `Resize((112, 112))`, `ToTensor()`, `Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`. Interpolation is torchvision default bilinear. No extra alignment step in the extractor.
- **Gallery (`*_Gallery_mcd.npz`):** `model.eval()`, **one** forward per image, L2-normalized `fe`. Not MC dropout.
- **Queries (`*_Queries_mcd.npz`):** `model.train()` so the backbone's built-in dropout is on; **`num_passes=40`** stochastic forwards per batch; `fe` is the **mean** embedding; `fu` is the mean per-dimension variance (scalar). The extractor does not set a dropout probability `p`; `p` is whatever the Hub snapshot uses. Default `--batch-size` is 32. `--seed` (default 0) now seeds torch, numpy, and random. **The paper cache predates that seed** and is not regenerated.
- **T1 query embeddings:** conformal scores use **`fe`** from `*_Queries_mcd.npz` (the 40-pass mean). Gallery-matched Method B scores `s_g` / `t_cal` come from `*_Queries_gmatch.npz` (`k=65`, seed 0). The adapter **Φ is full-gallery top-1 cosine**, not MC-dropout `fu`. T1 does not read `fu`.

DINOv2 caches (`*_Queries_wx.npz`, `cache/dinov2/pool_*_dinov2.npz`, `cache/dinov2/*_SCFace_Queries_dinov2.npz`) are `torch.hub` `dinov2_vitb14`, image size 112, `allow_upscale=False`, **eval / one forward** (not MC dropout). WRCP and DS-CP read these; WACR τ does not.

Gallery-matched scores: `k=65` identity-matched impostors, seed 0. KMM / KLIEP / domain-classifier weights: `*_Queries_kmm_median.npz`, `*_Queries_kliep.npz`, `*_Queries_wx.npz`.

## Score-level cache (`cache_share/`, gitignored)

`experiments/export_score_cache.py` writes one `.npz` per pair with calibration and SCFace query `t`, Method B scores against the matched and SCFace galleries, and baseline weights. No images and no embeddings. `run_all.py --from-scores` and `tests/test_reproduce.py --from-scores` recompute every in-scope item from that cache alone.

## Cache files (gitignored)

Each pair directory contains:

- `{Cal}_Queries_mcd.npz`, `{Cal}_Gallery_mcd.npz`
- `SCFace_Queries_mcd.npz`, `SCFace_Gallery_mcd.npz`
- `{Cal}_Queries_gmatch.npz` (gallery-matched top-1, k=65, seed 0)
- `{Cal}_Queries_kmm_median.npz`, `{Cal}_Queries_kliep.npz`, `{Cal}_Queries_wx.npz`

WRCP / DS-CP also read `cache/dinov2/` (pool and SCFace query DINOv2). Backbones: AdaFace ViT-Base WebFace4M (official Hub snapshot) and DINOv2 ViT-B/14 via `torch.hub`. Run `scripts/download_models.py`. Do not commit `.pth` / `.pt` files.
