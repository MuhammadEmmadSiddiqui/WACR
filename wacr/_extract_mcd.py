#!/usr/bin/env python3
"""
Monte Carlo Dropout (MCD) feature extraction with a fixed CVLFace AdaFace ViT backbone.

Pipeline
--------
1. Load ``minchul/cvlface_adaface_vit_base_webface4m`` (Hub or ``--model-dir`` snapshot).
2. Scan ``--input-dir`` for dataset folders (e.g. SCFace, TinyFace), each with Gallery/ and Queries/.
3. **Gallery:** ``eval()``, one forward per image batch → L2-normalized ``fe`` → ``.npz``.
4. **Queries:** ``train()`` (dropout on), ``num_passes`` forwards per batch → mean embedding ``fe``
   and per-dimension variance averaged to scalar ``fu`` → ``.npz``.

Outputs are written under ``cache/<combination-name>/``, e.g.
``cache/tinyface_scface/SCFace_Gallery_mcd.npz``. Downstream weight scripts
(``compute_kl_weights.py``, ``compute_kmm_weights.py``, etc.) read these caches.

Example (from repo root)::

  python Inference/extract_mcd_features.py --combination tinyface_scface
  python Inference/extract_mcd_features.py --combination tinyface_scface --batch-size 16 --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List, Literal, NamedTuple, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wacr.paths import (
    CACHE_DIR,
    REPO_ROOT,
    add_pair_argument,
    pair_datasets_dir,
    normalize_pair_name,
)

# =============================================================================
# Constants
# =============================================================================

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Only backbone supported by this script (Hub id).
BACKBONE_REPO_ID = "minchul/cvlface_adaface_vit_base_webface4m"

# Rough minimum size (bytes) so we treat ``pretrained_model/model.pt`` as real weights, not an LFS pointer.
MIN_WEIGHTS_FILE_BYTES = 10_000_000


class ExtractJob(NamedTuple):
    """Maps one image folder (under ``REPO_ROOT``) to one output ``.npz`` and a mode."""

    image_folder: str  # relative to REPO_ROOT, e.g. "datasets/tinyface_scface/SCFace/Gallery"
    output_npz: str  # filename only, e.g. "SCFace_Gallery_mcd.npz" under --output-dir/<combination>/
    mode: Literal["gallery", "query"]


SPLIT_TO_MODE: dict[str, Literal["gallery", "query"]] = {
    "Gallery": "gallery",
    "Queries": "query",
}


# =============================================================================
# Third-party imports required by Hub ``models/`` code (checked before download)
# =============================================================================


def ensure_hub_python_dependencies() -> None:
    """Exit with install hint if CVLFace Hub code cannot import its dependencies."""
    required: List[Tuple[str, str]] = [
        ("easydict", "easydict"),
        ("omegaconf", "omegaconf"),
        ("yaml", "PyYAML"),
    ]
    missing = [pip_name for import_name, pip_name in required if importlib.util.find_spec(import_name) is None]
    if not missing:
        return
    exe = sys.executable
    raise SystemExit(
        "Missing Python package(s) required by the Hugging Face CVLFace model code: "
        + ", ".join(missing)
        + "\n\nInstall into the SAME environment as this script (``pip`` alone may target another Python):\n\n"
        f"  {exe} -m pip install {' '.join(missing)}\n"
    )


# =============================================================================
# Images & tensor batches
# =============================================================================


def list_image_files(folder: Path) -> List[Path]:
    """All supported images under ``folder`` (recursive), sorted for stable order."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_transform() -> Compose:
    """AdaFace 112x112, then ToTensor + Normalize."""
    return Compose(
        [
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def discover_extract_jobs(input_dir: Path) -> List[ExtractJob]:
    """
    Build extraction jobs from a combination folder layout::

        <input-dir>/
          SCFace/Gallery/, SCFace/Queries/
          TinyFace/Gallery/, TinyFace/Queries/

    Jobs are ordered by dataset name; within each dataset, Gallery then Queries.
    """
    try:
        rel_base = input_dir.relative_to(REPO_ROOT)
    except ValueError as err:
        raise ValueError(
            f"--input-dir must lie under project root {REPO_ROOT}, got: {input_dir}"
        ) from err

    jobs: List[ExtractJob] = []
    for dataset_dir in sorted(input_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        for split_name, mode in SPLIT_TO_MODE.items():
            split_path = dataset_dir / split_name
            if not split_path.is_dir():
                continue
            rel_folder = f"{rel_base.as_posix()}/{dataset_name}/{split_name}"
            output_npz = f"{dataset_name}_{split_name}_mcd.npz"
            jobs.append(ExtractJob(rel_folder, output_npz, mode))

    if not jobs:
        raise FileNotFoundError(
            f"No Gallery/ or Queries/ folders found under dataset subdirs of {input_dir}"
        )
    return jobs


class ImagePathDataset(Dataset):
    """RGB image on disk → ``transform`` (tensor + norm) and absolute path string."""

    def __init__(self, paths: Sequence[Path], transform: Compose):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path.resolve())


def collate_image_batch(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    tensors = torch.stack([item[0] for item in batch], dim=0)
    paths = [item[1] for item in batch]
    return tensors, paths


def make_dataloader(
    paths: List[Path],
    transform: Compose,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        ImagePathDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_image_batch,
        drop_last=False,
    )


# =============================================================================
# Model: Hub layout, weights file, load
# =============================================================================


def unwrap_embedding_tensor(model_output: object) -> torch.Tensor:
    """CVLFace / HF may return a plain tensor or a small wrapper; normalize to (batch, dim)."""
    if torch.is_tensor(model_output):
        return model_output
    for attr in ("embeddings", "face_embedding", "logits", "pooler_output"):
        if hasattr(model_output, attr):
            t = getattr(model_output, attr)
            if t is not None and torch.is_tensor(t):
                return t[:, 0, :] if t.dim() == 3 else t
    if hasattr(model_output, "last_hidden_state"):
        t = model_output.last_hidden_state
        if torch.is_tensor(t) and t.dim() == 3:
            return t[:, 0, :]
    raise RuntimeError(f"Cannot interpret model output as (B, D) tensor: {type(model_output)}")


def embed_image_batch(model: torch.nn.Module, images_bchw: torch.Tensor) -> torch.Tensor:
    """One forward pass → (B, D) embedding tensor (not L2-normalized yet)."""
    return unwrap_embedding_tensor(model(images_bchw))


def _is_git_lfs_pointer_file(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def pretrained_weights_ready(path: Path) -> bool:
    """True if ``path`` looks like a full ``torch.load`` checkpoint, not missing / LFS stub."""
    if not path.is_file() or _is_git_lfs_pointer_file(path):
        return False
    return path.stat().st_size >= MIN_WEIGHTS_FILE_BYTES


def resolve_model_repo(local_dir: str | None) -> Path:
    """
    Directory that contains Hub layout: ``models/``, ``pretrained_model/``, ``wrapper.py``.

    Either ``--model-dir`` or a Hub snapshot downloaded under the default HF cache.
    """
    if local_dir:
        root = Path(local_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"--model-dir not found: {root}")
    else:
        root = Path(snapshot_download(repo_id=BACKBONE_REPO_ID, repo_type="model")).resolve()

    if not (root / "models").is_dir():
        raise FileNotFoundError(
            f"Expected a `models/` package under the model roo.  {root}\n"
            "Use --model-dir with a full snapshot of the backbone repo, or omit it to download from the Hub."
        )
    return root


def _copy_weights_from_hub_cache_to(destination_pt: Path) -> None:
    """Download ``pretrained_model/model.pt`` into HF cache, then copy bytes into ``destination_pt``."""
    print(
        "Materializing pretrained_model/model.pt from Hub cache (large on first run)...",
        flush=True,
    )
    cached_path = hf_hub_download(
        repo_id=BACKBONE_REPO_ID,
        filename="pretrained_model/model.pt",
        local_files_only=False,
    )
    src = Path(cached_path)
    if not src.is_file() or src.stat().st_size < 500_000:
        raise RuntimeError(f"Unexpected hf_hub_download result: {cached_path!r}")
    destination_pt.parent.mkdir(parents=True, exist_ok=True)
    if destination_pt.is_file():
        destination_pt.unlink()
    shutil.copy2(src, destination_pt)
    print(f"  -> {destination_pt} ({destination_pt.stat().st_size} bytes)", flush=True)


def _write_pt_from_root_safetensors(repo_root: Path, destination_pt: Path) -> None:
    safetensors_path = repo_root / "model.safetensors"
    if not safetensors_path.is_file() or safetensors_path.stat().st_size < 1_000_000:
        return
    try:
        from safetensors.torch import load_file
    except ImportError as err:
        raise SystemExit(
            "pretrained_model/model.pt is missing; model.safetensors exists but `safetensors` is not installed.\n"
            f"  {sys.executable} -m pip install safetensors\n"
            f"Original error: {err}"
        ) from err
    destination_pt.parent.mkdir(parents=True, exist_ok=True)
    state = load_file(str(safetensors_path))
    torch.save(state, str(destination_pt))
    print(f"Wrote {destination_pt} from model.safetensors ({len(state)} tensors).", flush=True)


def ensure_pretrained_model_pt(repo_root: Path) -> None:
    """
    CVLFace ``wrapper`` loads ``pretrained_model/model.pt`` with ``torch.load``.

    Snapshots sometimes ship an LFS pointer or omit the file; we copy from the Hub cache
    or rebuild from ``model.safetensors`` at repo root.
    """
    repo_root = repo_root.resolve()
    weights_pt = repo_root / "pretrained_model" / "model.pt"

    if pretrained_weights_ready(weights_pt):
        return

    if weights_pt.is_file():
        print(f"Replacing incomplete weights file: {weights_pt}", flush=True)
        weights_pt.unlink()

    try:
        _copy_weights_from_hub_cache_to(weights_pt)
    except Exception as err:  # noqa: BLE001 — best-effort then fallback
        print(f"hf_hub_download/copy failed: {err}", flush=True)

    if pretrained_weights_ready(weights_pt):
        return

    _write_pt_from_root_safetensors(repo_root, weights_pt)

    if not pretrained_weights_ready(weights_pt):
        raise FileNotFoundError(
            f"Could not obtain real weights at {weights_pt}.\n"
            f"Try: huggingface-cli download {BACKBONE_REPO_ID} --local-dir {repo_root}\n"
            "(or install git-lfs and re-pull the model snapshot.)"
        )


def patch_cvlface_load_state_dict_to_use_absolute_paths(repo_root_abs: str) -> None:
    """
    Hub code opens ``pretrained_model/model.pt`` relative to cwd.

    During ``AutoModel.from_pretrained`` the cwd may not stay on the repo; patch the
    loader so relative paths resolve under ``repo_root_abs``.
    """
    utils_mod = importlib.import_module("models.base.utils")
    base_mod = importlib.import_module("models.base")
    original_load = utils_mod.load_state_dict_from_path

    def load_state_dict_resolved(path: str | os.PathLike[str]) -> object:
        resolved = os.fspath(path)
        if not os.path.isabs(resolved):
            resolved = os.path.normpath(os.path.join(repo_root_abs, resolved))
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"Missing pretrained weights at {resolved!r}\n"
                f"  cwd={os.getcwd()!r}\n  repo_root={repo_root_abs!r}"
            )
        return original_load(resolved)

    utils_mod.load_state_dict_from_path = load_state_dict_resolved
    base_mod.load_state_dict_from_path = load_state_dict_resolved


def load_backbone(device: torch.device, local_dir: str | None) -> torch.nn.Module:
    """Download or use local snapshot, fix weights path, then ``AutoModel.from_pretrained``."""
    repo = resolve_model_repo(local_dir)
    ensure_pretrained_model_pt(repo)

    repo_abs = os.path.abspath(str(repo))
    if repo_abs not in sys.path:
        sys.path.insert(0, repo_abs)
    patch_cvlface_load_state_dict_to_use_absolute_paths(repo_abs)

    previous_cwd = os.getcwd()
    try:
        os.chdir(repo_abs)
        model = AutoModel.from_pretrained(repo_abs, trust_remote_code=True, local_files_only=True)
    finally:
        os.chdir(previous_cwd)

    model.eval()
    return model.to(device)


# =============================================================================
# Inference: gallery (deterministic) vs queries (MCD)
# =============================================================================


def _existing_npz_ok(out_path: Path, n_images: int) -> bool:
    if not out_path.is_file() or n_images < 0:
        return False
    try:
        with np.load(out_path, allow_pickle=True) as z:
            n = int(np.asarray(z["fe"]).shape[0])
    except (OSError, KeyError, ValueError):
        return False
    return n == n_images


@torch.no_grad()
def run_gallery_pass(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """eval(), one forward per batch, L2-normalized rows → numpy ``fe`` + path list."""
    model.eval()
    embedding_chunks: List[torch.Tensor] = []
    path_list: List[str] = []
    n_batches = len(loader)
    t0 = time.perf_counter()
    for bi, (images, paths) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        embeddings = embed_image_batch(model, images)
        normalized = F.normalize(embeddings.float(), dim=-1)
        embedding_chunks.append(normalized.cpu())
        path_list.extend(paths)
        if bi == 1 or bi == n_batches or bi % 20 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  gallery batch {bi}/{n_batches}  elapsed={elapsed:.0f}s", flush=True)

    if not embedding_chunks:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=object)

    fe = torch.cat(embedding_chunks, dim=0).numpy().astype(np.float32)
    paths_arr = np.array(path_list, dtype=object)
    return fe, paths_arr


@torch.no_grad()
def run_query_mcd_passes(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_passes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    train() so dropout stays random; ``num_passes`` forwards per batch.

    Each pass is L2-normalized before stacking. fe: L2(mean over passes).
    fu: mean over embedding dims of variance across normalized passes.
    """
    if num_passes < 1:
        raise ValueError("num_passes must be >= 1")

    model.train()
    fe_chunks: List[torch.Tensor] = []
    fu_chunks: List[torch.Tensor] = []
    path_list: List[str] = []
    n_batches = len(loader)
    t0 = time.perf_counter()

    for bi, (images, paths) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        # Stack shape [num_passes, batch, dim]
        stochastic_embeddings: List[torch.Tensor] = []
        for _ in range(num_passes):
            raw_emb = embed_image_batch(model, images).float()
            norm_emb = F.normalize(raw_emb, dim=-1)
            stochastic_embeddings.append(norm_emb)
        stacked = torch.stack(stochastic_embeddings, dim=0)

        fe_mean = stacked.mean(dim=0)
        fe_out = F.normalize(fe_mean, dim=-1)
        variance_across_passes = stacked.var(dim=0, correction=0)
        fu_out = variance_across_passes.mean(dim=-1)

        fe_chunks.append(fe_out.cpu())
        fu_chunks.append(fu_out.cpu())
        path_list.extend(paths)
        elapsed = time.perf_counter() - t0
        eta = (elapsed / bi) * (n_batches - bi) if bi else 0.0
        print(
            f"  query MCD batch {bi}/{n_batches}  passes={num_passes}  "
            f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
            flush=True,
        )

    if not fe_chunks:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.array([], dtype=object),
        )

    fe = torch.cat(fe_chunks, dim=0).numpy().astype(np.float32)
    fu = torch.cat(fu_chunks, dim=0).numpy().astype(np.float32)
    paths_arr = np.array(path_list, dtype=object)
    return fe, fu, paths_arr


# =============================================================================
# Saving & per-job orchestration
# =============================================================================


def save_gallery_npz(out_path: Path, fe: np.ndarray, paths: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, fe=fe, paths=paths)
    print(f"  Wrote {out_path.resolve()}  fe.shape={fe.shape}", flush=True)


def save_query_npz(out_path: Path, fe: np.ndarray, fu: np.ndarray, paths: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, fe=fe, fu=fu, paths=paths)
    print(f"  Wrote {out_path.resolve()}  fe.shape={fe.shape}  fu.shape={fu.shape}", flush=True)


def write_empty_result_npz(out_path: Path, mode: Literal["gallery", "query"]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "gallery":
        np.savez_compressed(
            out_path,
            fe=np.zeros((0, 512), dtype=np.float32),
            paths=np.array([], dtype=object),
        )
    else:
        np.savez_compressed(
            out_path,
            fe=np.zeros((0, 512), dtype=np.float32),
            fu=np.zeros((0,), dtype=np.float32),
            paths=np.array([], dtype=object),
        )
    print(f"  Wrote empty archive {out_path.resolve()}", flush=True)


def run_extract_job(
    job: ExtractJob,
    *,
    model: torch.nn.Module,
    transform: Compose,
    out_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_passes: int,
    skip_existing: bool = False,
) -> None:
    image_dir = REPO_ROOT / job.image_folder
    image_paths = list_image_files(image_dir)
    out_path = out_root / job.output_npz

    print(f"\n=== {job.image_folder} -> {job.output_npz} ({job.mode}, n={len(image_paths)}) ===", flush=True)

    if skip_existing and _existing_npz_ok(out_path, len(image_paths)):
        print(f"  skip existing {out_path.resolve()}", flush=True)
        return

    if not image_paths:
        print("  Skipping: no images.", flush=True)
        write_empty_result_npz(out_path, job.mode)
        return

    loader = make_dataloader(
        image_paths,
        transform,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    if job.mode == "gallery":
        fe, paths_arr = run_gallery_pass(model, loader, device)
        save_gallery_npz(out_path, fe, paths_arr)
    elif job.mode == "query":
        fe, fu, paths_arr = run_query_mcd_passes(model, loader, device, num_passes)
        save_query_npz(out_path, fe, fu, paths_arr)
    else:
        raise ValueError(f"Unknown mode: {job.mode}")


# =============================================================================
# CLI
# =============================================================================


def set_extract_seed(seed: int) -> None:
    """Seed torch, numpy, and random. Does not rewrite an existing cache."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MCD feature extraction: scan a combination folder (datasets per Gallery/Queries) "
            "and write one .npz per split under cache/<combination-name>/."
        )
    )
    add_pair_argument(
        parser,
        help_text=(
            "Pair folder name under data/images/, e.g. tinyface_scface. "
            "Must contain subfolders (SCFace, TinyFace, …) each with Gallery/ and Queries/."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override the image root (default: data/images/<pair>/).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help=(
            "Local Hugging Face snapshot of minchul/cvlface_adaface_vit_base_webface4m "
            "(same layout as the Hub). If omitted, that repo is downloaded from the Hub."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CACHE_DIR),
        help="Base directory for *_mcd.npz files (default: cache/).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-passes",
        type=int,
        default=40,
        help="Stochastic forward passes per query batch (MCD only).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="e.g. cuda or cuda:0")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a split if the output npz already exists with the expected number of rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for torch, numpy, and random (query dropout). Default 0. "
        "The paper cache predates this flag and is not regenerated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_hub_python_dependencies()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is False.")
    if device.type == "cpu":
        n_threads = max(1, int(os.cpu_count() or 1))
        torch.set_num_threads(n_threads)
        print(f"CPU threads: {n_threads}", flush=True)

    set_extract_seed(args.seed)
    combination = normalize_pair_name(args.pair)
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else pair_datasets_dir(combination)
    jobs = discover_extract_jobs(input_dir)
    out_root = Path(args.output_dir).expanduser().resolve() / combination
    transform = image_transform()

    print(f"Device: {device}", flush=True)
    print(
        f"Backbone: {BACKBONE_REPO_ID}"
        + (f" (local snapshot: {args.model_dir})" if args.model_dir else " (from Hub cache)"),
        flush=True,
    )
    print(f"Pair: {combination}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Input dir:  {input_dir}", flush=True)
    print(f"Output dir: {out_root}", flush=True)
    print(f"Jobs ({len(jobs)}):", flush=True)
    for job in jobs:
        print(f"  {job.image_folder} -> {job.output_npz} ({job.mode})", flush=True)

    model = load_backbone(device, args.model_dir)

    for job in jobs:
        run_extract_job(
            job,
            model=model,
            transform=transform,
            out_root=out_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            num_passes=args.num_passes,
            skip_existing=args.skip_existing,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
