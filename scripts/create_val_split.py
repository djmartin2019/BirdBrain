"""
Create a stratified train/val split from CUB's official TRAIN images.

Official CUB test images (is_train=0) are never included here — they stay
held out for final evaluation only.

Usage (from project root):
    python scripts/create_val_split.py
    python scripts/create_val_split.py --ratio 0.1 --seed 42
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/raw/CUB_200_2011"
DEFAULT_OUTPUT = PROJECT_ROOT / "splits/val_split.txt"


def create_val_split(data_dir: Path, output_path: Path, ratio: float, seed: int):
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    images = pd.read_csv(data_dir / "images.txt", sep=" ", names=["image_id", "filepath"])
    labels = pd.read_csv(
        data_dir / "image_class_labels.txt",
        sep=" ",
        names=["image_id", "class_id"],
    )
    official = pd.read_csv(
        data_dir / "train_test_split.txt",
        sep=" ",
        names=["image_id", "is_train"],
    )

    df = images.merge(labels, on="image_id").merge(official, on="image_id")
    train_pool = df[df["is_train"] == 1].copy()

    _, val_df = train_test_split(
        train_pool,
        test_size=ratio,
        random_state=seed,
        stratify=train_pool["class_id"],
    )

    val_ids = set(val_df["image_id"])
    train_pool["is_val"] = train_pool["image_id"].isin(val_ids).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_pool[["image_id", "is_val"]].to_csv(
        output_path,
        sep=" ",
        header=False,
        index=False,
    )

    n_val = int(train_pool["is_val"].sum())
    n_train = len(train_pool) - n_val
    n_classes_val = val_df["class_id"].nunique()

    print(f"Wrote {output_path}")
    print(f"  train images: {n_train}")
    print(f"  val images:   {n_val}")
    print(f"  val classes:  {n_classes_val} / {train_pool['class_id'].nunique()}")
    print(f"  ratio:        {ratio}, seed: {seed}")

    assert train_pool["image_id"].isin(val_df["image_id"]).sum() == len(val_df)
    assert n_classes_val == train_pool["class_id"].nunique(), "Every class must appear in val"


def main():
    parser = argparse.ArgumentParser(description="Create CUB train/val split file")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ratio", type=float, default=0.1, help="Fraction of official train for val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_val_split(args.data_dir, args.output, args.ratio, args.seed)


if __name__ == "__main__":
    main()
