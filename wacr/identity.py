"""Identity extraction from image paths."""

from __future__ import annotations

from pathlib import Path

SCFACE_DISTANCES = (1, 2, 3)


def identity_of(path_str: str) -> str:
    """Identity = stem before the first underscore.

    Works for TinyFace ``1556_2.jpg`` -> ``1556`` and SCFace
    ``066_cam1_1.jpg`` -> ``066``.
    """
    return Path(path_str).stem.split("_", 1)[0]


def scface_distance_of(path_str: str) -> int:
    """SCFace capture distance = last underscore segment of the stem.

    ``066_cam1_1.jpg`` -> ``1`` and ``066_cam1_3.jpg`` -> ``3``. SCFace has
    three webcam distances; raises ``ValueError`` for stems that do not end in
    an integer in :data:`SCFACE_DISTANCES`.
    """
    last = Path(path_str).stem.rsplit("_", 1)[-1]
    try:
        distance = int(last)
    except ValueError:
        raise ValueError(
            f"Cannot parse SCFace distance from {path_str!r}: trailing segment "
            f"{last!r} is not an integer."
        ) from None
    if distance not in SCFACE_DISTANCES:
        raise ValueError(
            f"SCFace distance {distance} from {path_str!r} is not one of "
            f"{SCFACE_DISTANCES}."
        )
    return distance


# Alias used in WACR and Retrieval/retrieval_eval.py.
identity_before_first_underscore = identity_of
