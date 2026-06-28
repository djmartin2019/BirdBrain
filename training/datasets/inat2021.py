import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

AVES_CLASS = "Aves"
NUM_CLASSES = 1486


def build_bird_category_map(categories: list) -> dict[int, int]:
    """Map iNat category_id to contiguous bird labels 0..NUM_CLASSES-1."""
    bird_cats = sorted(
        [c for c in categories if c.get("class") == AVES_CLASS],
        key=lambda c: c["id"],
    )
    return {c["id"]: idx for idx, c in enumerate(bird_cats)}


def resolve_image_path(root_dir: Path, file_name: str) -> Path | None:
    """Resolve a COCO file_name to an on-disk path."""
    root_dir = Path(root_dir)
    candidates = [root_dir / file_name]
    if file_name.startswith("train_mini/"):
        rest = file_name[len("train_mini/") :]
        candidates.append(root_dir / "train_mini" / "train_mini" / rest)

    for path in candidates:
        if path.exists():
            return path
    return None


def load_bird_samples(root_dir: Path, annotation_file: Path) -> pd.DataFrame:
    """Load bird-only samples from a COCO JSON annotation file."""
    with open(annotation_file) as f:
        coco = json.load(f)

    cat_map = build_bird_category_map(coco["categories"])

    images_df = pd.DataFrame(coco["images"]).rename(columns={"id": "image_id"})
    ann_df = pd.DataFrame(coco["annotations"])
    ann_df = ann_df[ann_df["category_id"].isin(cat_map)]
    ann_df["label"] = ann_df["category_id"].map(cat_map)

    return images_df.merge(ann_df[["image_id", "label"]], on="image_id", how="inner")


class INat2021Dataset(Dataset):
    """Birds-only (Aves) subset of iNaturalist 2021 in COCO JSON format."""

    NUM_CLASSES = NUM_CLASSES

    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        val_split_file=None,
        skip_missing=True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        if split == "test":
            annotation_file = self.root_dir / "val.json"
            df = load_bird_samples(self.root_dir, annotation_file)
        elif split in ("train", "val"):
            annotation_file = self.root_dir / "train_mini.json"
            df = load_bird_samples(self.root_dir, annotation_file)

            val_split_path = Path(val_split_file) if val_split_file else None
            if val_split_path is None:
                val_split_path = self.root_dir / "inat_val_split.txt"

            if not val_split_path.exists():
                raise FileNotFoundError(
                    f"Validation split file not found: {val_split_path}\n"
                    "Run: python scripts/create_inat_val_split.py"
                )

            val_split = pd.read_csv(
                val_split_path,
                sep=" ",
                names=["image_id", "is_val"],
            )
            df = df.merge(val_split, on="image_id", how="left")
            df["is_val"] = df["is_val"].fillna(0).astype(int)

            if split == "train":
                df = df[df["is_val"] == 0]
            else:
                df = df[df["is_val"] == 1]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        df = df.reset_index(drop=True)
        df["image_path"] = df["file_name"].apply(
            lambda fn: resolve_image_path(self.root_dir, fn)
        )

        total = len(df)
        if skip_missing:
            df = df[df["image_path"].notna()].reset_index(drop=True)
            print(
                f"INat2021Dataset ({split}): kept {len(df)} / {total} images "
                f"({total - len(df)} missing on disk)"
            )
        else:
            missing = df[df["image_path"].isna()]
            if not missing.empty:
                first = missing.iloc[0]["file_name"]
                raise FileNotFoundError(
                    f"Missing {len(missing)} images; first missing: {first}"
                )

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
