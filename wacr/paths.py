"""Repository and cache path helpers."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cache"
DATA_DIR = REPO_ROOT / "data"
DATASETS_DIR = DATA_DIR / "images"
DEFAULT_CALIB_DATASET = "TinyFace"
DEFAULT_TEST_DATASET = "SCFace"

PAIR_ALIASES = {
    "tinyface_scface": "tinyface_scface",
    "xqlfw_scface": "xqlfw_scface",
    "TinyFace": "tinyface_scface",
    "XQLFW": "xqlfw_scface",
}


def normalize_pair_name(name: str) -> str:
    s = name.strip()
    if s in PAIR_ALIASES:
        return PAIR_ALIASES[s]
    return s


def pair_cache_dir(pair: str) -> Path:
    return (CACHE_DIR / normalize_pair_name(pair)).resolve()


def pair_datasets_dir(pair: str) -> Path:
    path = (DATASETS_DIR / normalize_pair_name(pair)).resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {path}\n"
            f"Expected layout: data/images/{normalize_pair_name(pair)}/{{SCFace,TinyFace,XQLFW}}/{{Gallery,Queries}}/"
        )
    return path


def mcd_queries_npz(pair: str, dataset: str) -> Path:
    return pair_cache_dir(pair) / f"{dataset}_Queries_mcd.npz"


def weight_output_npz(pair: str, cal_dataset: str, weight_kind: str) -> Path:
    return pair_cache_dir(pair) / f"{cal_dataset}_Queries_{weight_kind}.npz"


def resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def resolve_input_dir(path_str: str) -> Path:
    p = resolve_path(path_str)
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")
    return p


def add_pair_argument(parser, *, help_text: str | None = None) -> None:
    parser.add_argument(
        "--pair",
        type=str,
        required=True,
        help=help_text or "Pair folder name, e.g. tinyface_scface.",
    )


def add_combination_argument(parser, *, help_text: str | None = None) -> None:
    parser.add_argument(
        "--combination",
        dest="pair",
        type=str,
        required=True,
        help=help_text or "Pair folder name, e.g. tinyface_scface.",
    )


def resolve_weight_script_paths(
    pair: str,
    *,
    weight_kind: str,
    cal_dataset: str = DEFAULT_CALIB_DATASET,
    test_dataset: str = DEFAULT_TEST_DATASET,
    calib: str | None = None,
    test_npz: str | None = None,
    output: str | None = None,
) -> tuple[Path, Path, Path]:
    calib_path = resolve_path(calib) if calib else mcd_queries_npz(pair, cal_dataset)
    test_path = resolve_path(test_npz) if test_npz else mcd_queries_npz(pair, test_dataset)
    output_path = resolve_path(output) if output else weight_output_npz(pair, cal_dataset, weight_kind)
    return calib_path, test_path, output_path


def canonical_path_key(p: str) -> str:
    return str(Path(p).as_posix())


# Aliases so copied scripts keep their import names.
combination_cache_dir = pair_cache_dir
combination_datasets_dir = pair_datasets_dir
normalize_combination_name = normalize_pair_name
EMBEDDING_CACHE = CACHE_DIR
TRAIN_DIR = REPO_ROOT / "baselines"
RETRIEVAL_DIR = REPO_ROOT / "data"
