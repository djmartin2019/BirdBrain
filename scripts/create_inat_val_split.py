"""
Create a stratified train/val split from iNat 2021 train_mini bird images.

Official val.json (test split) is never included here — it stays held out for
final evaluation only.

Usage (from project root):
    python scripts/create_inat_val_split.py
    python scripts/create_inat_val_split.py --ratio 0.1 --seed 42
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/raw/iNat_2021"
DEFAULT_OUTPUT = PROJECT_ROOT / "splits/inat_val_split.txt"

sys.path.insert(0, str(PROJECT_ROOT / "training"))
from datasets.inat2021 import load_bird_samples  # noqa: E402


def create_inat_val_split(data_dir: Path, output_path: Path, ratio: float, seed: int):
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    annotation_file = data_dir / "train_mini.json"
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    df = load_bird_samples(data_dir, annotation_file)

    _, val_df = train_test_split(
        df,
        test_size=ratio,
        random_state=seed,
        stratify=df["label"],
    )

    val_ids = set(val_df["image_id"])
    df["is_val"] = df["image_id"].isin(val_ids).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[["image_id", "is_val"]].to_csv(
        output_path,
        sep=" ",
        header=False,
        index=False,
    )

    n_val = int(df["is_val"].sum())
    n_train = len(df) - n_val
    n_classes_val = val_df["label"].nunique()

    print(f"Wrote {output_path}")
    print(f"  bird train images: {n_train}")
    print(f"  bird val images:   {n_val}")
    print(f"  val classes:       {n_classes_val} / {df['label'].nunique()}")
    print(f"  ratio:             {ratio}, seed: {seed}")

    assert df["image_id"].isin(val_df["image_id"]).sum() == len(val_df)
    assert n_classes_val == df["label"].nunique(), "Every class must appear in val"


def main():
    parser = argparse.ArgumentParser(
        description="Create iNat 2021 train/val split file (birds only)"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="Fraction of train_mini bird images for val",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_inat_val_split(args.data_dir, args.output, args.ratio, args.seed)


if __name__ == "__main__":
    main()
