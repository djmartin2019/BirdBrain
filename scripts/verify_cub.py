"""
Verify CUB-200-2011 dataset layout, image counts, val split, and loader smoke test.

Usage (from project root):
    python scripts/verify_cub.py
    python scripts/verify_cub.py --data-dir data/raw/CUB_200_2011 --val-split splits/val_split.txt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/raw/CUB_200_2011"
DEFAULT_VAL_SPLIT = PROJECT_ROOT / "splits/val_split.txt"

REQUIRED_FILES = (
    "images.txt",
    "classes.txt",
    "image_class_labels.txt",
    "train_test_split.txt",
    "bounding_boxes.txt",
)

sys.path.insert(0, str(PROJECT_ROOT / "training"))
from datasets.cub import CUBDataset  # noqa: E402


def _check(name: str, ok: bool, detail: str, failures: list) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not ok:
        failures.append(name)


def verify_cub(data_dir: Path, val_split: Path) -> list[str]:
    data_dir = Path(data_dir)
    val_split = Path(val_split)
    failures: list[str] = []

    for filename in REQUIRED_FILES:
        path = data_dir / filename
        _check(
            f"file {filename}",
            path.exists(),
            str(path) if path.exists() else "missing",
            failures,
        )

    images_txt = data_dir / "images.txt"
    if images_txt.exists():
        images = pd.read_csv(images_txt, sep=" ", names=["image_id", "filepath"])
        expected = len(images)
        jpgs = list((data_dir / "images").rglob("*.jpg"))
        _check(
            "image count",
            len(jpgs) == expected,
            f"{len(jpgs)} on disk, {expected} in images.txt",
            failures,
        )

        official = pd.read_csv(
            data_dir / "train_test_split.txt",
            sep=" ",
            names=["image_id", "is_train"],
        )
        train_pool = len(official[official["is_train"] == 1])
    else:
        train_pool = None

    if val_split.exists():
        split_rows = sum(1 for _ in open(val_split))
        if train_pool is not None:
            _check(
                "val split rows",
                split_rows == train_pool,
                f"{split_rows} rows, {train_pool} official train images",
                failures,
            )
        else:
            _check("val split rows", split_rows > 0, f"{split_rows} rows", failures)
    else:
        _check("val split file", False, f"missing {val_split}", failures)

    if not failures:
        for split in ("train", "val", "test"):
            try:
                ds = CUBDataset(data_dir, split=split, val_split_file=val_split)
                _check(
                    f"loader {split}",
                    len(ds) > 0,
                    f"{len(ds)} samples",
                    failures,
                )
            except Exception as exc:
                _check(f"loader {split}", False, str(exc), failures)

    return failures


def main():
    parser = argparse.ArgumentParser(description="Verify CUB-200-2011 dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--val-split", type=Path, default=DEFAULT_VAL_SPLIT)
    args = parser.parse_args()

    print(f"Verifying CUB dataset at {args.data_dir}")
    failures = verify_cub(args.data_dir, args.val_split)

    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)

    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
