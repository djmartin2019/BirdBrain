#!/usr/bin/env python3
"""Export CUB-200 species names for the web frontend.

Reads api/labels.json (generated from CUB classes.txt) and writes
web/src/lib/species.json as a sorted-friendly ordered list.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = PROJECT_ROOT / "api" / "labels.json"
OUTPUT_PATH = PROJECT_ROOT / "web" / "src" / "lib" / "species.json"
CLASSES_PATH = PROJECT_ROOT / "data" / "raw" / "CUB_200_2011" / "classes.txt"


def load_labels() -> dict[int, str]:
    if LABELS_PATH.exists() and LABELS_PATH.stat().st_size > 0:
        with open(LABELS_PATH) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"No labels at {LABELS_PATH} and no CUB classes at {CLASSES_PATH}"
        )

    labels: dict[int, str] = {}
    for line in CLASSES_PATH.read_text().strip().splitlines():
        class_id, name = line.split(" ", 1)
        labels[int(class_id) - 1] = name.split(".", 1)[1].replace("_", " ")
    return labels


def main():
    labels = load_labels()
    species = [labels[i] for i in range(len(labels))]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(species, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(species)} species to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
