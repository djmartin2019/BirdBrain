#!/usr/bin/env python3
"""Export merged species names for the web frontend.

Reads api/labels.json (CUB-200) and api/inat_labels.json (iNat birds),
deduplicates by normalized name (CUB names win on overlap), and writes
web/src/lib/species.json as a sorted list.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUB_LABELS_PATH = PROJECT_ROOT / "api" / "labels.json"
INAT_LABELS_PATH = PROJECT_ROOT / "api" / "inat_labels.json"
OUTPUT_PATH = PROJECT_ROOT / "web" / "src" / "lib" / "species.json"
CLASSES_PATH = PROJECT_ROOT / "data" / "raw" / "CUB_200_2011" / "classes.txt"


def normalize_name(name: str) -> str:
    """Lowercase key for deduplication; strip punctuation and extra spaces."""
    key = name.lower()
    key = re.sub(r"[^\w\s]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def load_cub_labels() -> dict[int, str]:
    if CUB_LABELS_PATH.exists() and CUB_LABELS_PATH.stat().st_size > 0:
        with open(CUB_LABELS_PATH) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"No labels at {CUB_LABELS_PATH} and no CUB classes at {CLASSES_PATH}"
        )

    labels: dict[int, str] = {}
    for line in CLASSES_PATH.read_text().strip().splitlines():
        class_id, name = line.split(" ", 1)
        labels[int(class_id) - 1] = name.split(".", 1)[1].replace("_", " ")
    return labels


def load_inat_labels() -> dict[int, str]:
    if not INAT_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INAT_LABELS_PATH}. Run: python scripts/make_inat_labels.py"
        )
    with open(INAT_LABELS_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def merge_species(cub: dict[int, str], inat: dict[int, str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []

    for i in range(len(cub)):
        name = cub[i]
        key = normalize_name(name)
        if key not in seen:
            seen.add(key)
            merged.append(name)

    for i in range(len(inat)):
        name = inat[i]
        key = normalize_name(name)
        if key not in seen:
            seen.add(key)
            merged.append(name)

    merged.sort(key=lambda n: n.lower())
    return merged


def main():
    cub = load_cub_labels()
    inat = load_inat_labels()
    species = merge_species(cub, inat)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(species, f, indent=2)
        f.write("\n")
    print(
        f"Wrote {len(species)} species to {OUTPUT_PATH} "
        f"({len(cub)} CUB + {len(inat)} iNat, deduplicated)"
    )


if __name__ == "__main__":
    main()
