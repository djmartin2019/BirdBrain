"""
Verify iNaturalist 2021 train_mini + val layout, bird subset, and loader smoke test.

Usage (from project root):
    python scripts/verify_inat2021.py
    python scripts/verify_inat2021.py --data-dir data/raw/iNat_2021 --val-split splits/inat_val_split.txt
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/raw/iNat_2021"
DEFAULT_VAL_SPLIT = PROJECT_ROOT / "splits/inat_val_split.txt"

EXPECTED_TRAIN_MINI_IMAGES = 500_000
EXPECTED_VAL_IMAGES = 100_000
EXPECTED_CLASS_DIRS = 10_000
EXPECTED_BIRD_DIRS = 1_486
EXPECTED_TRAIN_MINI_BIRDS = 74_300
EXPECTED_VAL_BIRDS = 14_860
EXPECTED_VAL_SPLIT_ROWS = 74_300
EXPECTED_LOADER_TRAIN = 66_870
EXPECTED_LOADER_VAL = 7_430
EXPECTED_LOADER_TEST = 14_860

sys.path.insert(0, str(PROJECT_ROOT / "training"))
from datasets.inat2021 import (  # noqa: E402
    INat2021Dataset,
    load_bird_samples,
    resolve_image_path,
)


def _check(name: str, ok: bool, detail: str, failures: list) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not ok:
        failures.append(name)


def _count_class_dirs(root: Path) -> int:
    return sum(1 for p in root.iterdir() if p.is_dir())


def _count_bird_dirs(root: Path) -> int:
    return sum(1 for p in root.iterdir() if p.is_dir() and "_Aves_" in p.name)


def _count_jpgs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.jpg"))


def _json_resolution(data_dir: Path, json_path: Path) -> tuple[int, int]:
    with open(json_path) as f:
        coco = json.load(f)
    total = len(coco["images"])
    found = sum(
        1 for img in coco["images"] if resolve_image_path(data_dir, img["file_name"])
    )
    return found, total


def verify_inat2021(data_dir: Path, val_split: Path) -> list[str]:
    data_dir = Path(data_dir)
    val_split = Path(val_split)
    failures: list[str] = []

    for name in ("train_mini.json", "val.json"):
        _check(
            f"file {name}",
            (data_dir / name).exists(),
            str(data_dir / name),
            failures,
        )

    for name in ("train_mini", "val"):
        path = data_dir / name
        _check(f"dir {name}/", path.is_dir(), str(path), failures)

    train_mini_dir = data_dir / "train_mini"
    val_dir = data_dir / "val"
    if train_mini_dir.is_dir():
        _check(
            "train_mini class dirs",
            _count_class_dirs(train_mini_dir) == EXPECTED_CLASS_DIRS,
            f"{_count_class_dirs(train_mini_dir)} (expected {EXPECTED_CLASS_DIRS})",
            failures,
        )
        _check(
            "train_mini bird dirs",
            _count_bird_dirs(train_mini_dir) == EXPECTED_BIRD_DIRS,
            f"{_count_bird_dirs(train_mini_dir)} (expected {EXPECTED_BIRD_DIRS})",
            failures,
        )
        _check(
            "train_mini images",
            _count_jpgs(train_mini_dir) == EXPECTED_TRAIN_MINI_IMAGES,
            f"{_count_jpgs(train_mini_dir)} (expected {EXPECTED_TRAIN_MINI_IMAGES})",
            failures,
        )

    if val_dir.is_dir():
        _check(
            "val class dirs",
            _count_class_dirs(val_dir) == EXPECTED_CLASS_DIRS,
            f"{_count_class_dirs(val_dir)} (expected {EXPECTED_CLASS_DIRS})",
            failures,
        )
        _check(
            "val images",
            _count_jpgs(val_dir) == EXPECTED_VAL_IMAGES,
            f"{_count_jpgs(val_dir)} (expected {EXPECTED_VAL_IMAGES})",
            failures,
        )

    train_json = data_dir / "train_mini.json"
    val_json = data_dir / "val.json"
    if train_json.exists():
        found, total = _json_resolution(data_dir, train_json)
        _check(
            "train_mini.json paths",
            found == total,
            f"{found}/{total} resolve",
            failures,
        )
        birds = load_bird_samples(data_dir, train_json)
        bird_found = sum(
            1 for _, row in birds.iterrows() if resolve_image_path(data_dir, row["file_name"])
        )
        _check(
            "train_mini bird entries",
            len(birds) == EXPECTED_TRAIN_MINI_BIRDS,
            f"{len(birds)} in JSON (expected {EXPECTED_TRAIN_MINI_BIRDS})",
            failures,
        )
        _check(
            "train_mini bird paths",
            bird_found == EXPECTED_TRAIN_MINI_BIRDS,
            f"{bird_found}/{len(birds)} resolve",
            failures,
        )

    if val_json.exists():
        found, total = _json_resolution(data_dir, val_json)
        _check(
            "val.json paths",
            found == total,
            f"{found}/{total} resolve",
            failures,
        )
        birds = load_bird_samples(data_dir, val_json)
        bird_found = sum(
            1 for _, row in birds.iterrows() if resolve_image_path(data_dir, row["file_name"])
        )
        _check(
            "val bird entries",
            len(birds) == EXPECTED_VAL_BIRDS,
            f"{len(birds)} in JSON (expected {EXPECTED_VAL_BIRDS})",
            failures,
        )
        _check(
            "val bird paths",
            bird_found == EXPECTED_VAL_BIRDS,
            f"{bird_found}/{len(birds)} resolve",
            failures,
        )

    if val_split.exists():
        split_rows = sum(1 for _ in open(val_split))
        _check(
            "val split rows",
            split_rows == EXPECTED_VAL_SPLIT_ROWS,
            f"{split_rows} (expected {EXPECTED_VAL_SPLIT_ROWS})",
            failures,
        )
    else:
        _check("val split file", False, f"missing {val_split}", failures)

    if not failures:
        expected_sizes = {
            "train": EXPECTED_LOADER_TRAIN,
            "val": EXPECTED_LOADER_VAL,
            "test": EXPECTED_LOADER_TEST,
        }
        for split, expected in expected_sizes.items():
            try:
                ds = INat2021Dataset(
                    data_dir,
                    split=split,
                    val_split_file=val_split,
                    skip_missing=False,
                )
                _check(
                    f"loader {split}",
                    len(ds) == expected,
                    f"{len(ds)} samples (expected {expected})",
                    failures,
                )
            except Exception as exc:
                _check(f"loader {split}", False, str(exc), failures)

    return failures


def main():
    parser = argparse.ArgumentParser(description="Verify iNaturalist 2021 dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--val-split", type=Path, default=DEFAULT_VAL_SPLIT)
    args = parser.parse_args()

    print(f"Verifying iNat 2021 dataset at {args.data_dir}")
    failures = verify_inat2021(args.data_dir, args.val_split)

    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)

    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
