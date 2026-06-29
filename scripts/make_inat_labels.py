#!/usr/bin/env python3
"""Generate api/inat_labels.json from iNat 2021 train_mini.json bird categories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/raw/iNat_2021"
DEFAULT_OUTPUT = PROJECT_ROOT / "api/inat_labels.json"
AVES_CLASS = "Aves"
NUM_CLASSES = 1486


def build_bird_category_map(categories: list) -> dict[int, int]:
    bird_cats = sorted(
        [c for c in categories if c.get("class") == AVES_CLASS],
        key=lambda c: c["id"],
    )
    return {c["id"]: idx for idx, c in enumerate(bird_cats)}


def build_labels(data_dir: Path) -> dict[int, str]:
    annotation_file = data_dir / "train_mini.json"
    if not annotation_file.exists():
        raise FileNotFoundError(f"Missing iNat annotations: {annotation_file}")

    with open(annotation_file) as f:
        coco = json.load(f)

    cat_map = build_bird_category_map(coco["categories"])
    id_to_cat = {c["id"]: c for c in coco["categories"] if c.get("class") == AVES_CLASS}

    labels: dict[int, str] = {}
    for category_id, label_idx in sorted(cat_map.items(), key=lambda item: item[1]):
        cat = id_to_cat[category_id]
        name = (cat.get("common_name") or cat.get("name") or str(label_idx)).strip()
        labels[label_idx] = name

    if len(labels) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} bird labels, got {len(labels)}")

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="iNat 2021 root containing train_mini.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output labels JSON path",
    )
    args = parser.parse_args()

    labels = build_labels(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(labels)} labels to {args.output}")


if __name__ == "__main__":
    main()
