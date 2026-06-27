from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def crop_to_bbox(image, x, y, width, height):
    """Crop image to CUB bounding box, clamped to image bounds."""
    img_w, img_h = image.size
    left = max(0, int(x))
    top = max(0, int(y))
    right = min(img_w, int(x + width))
    bottom = min(img_h, int(y + height))

    if right <= left or bottom <= top:
        return image

    return image.crop((left, top, right, bottom))


class CUBDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, use_bbox_crop=False):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.transform = transform
        self.use_bbox_crop = use_bbox_crop

        images = pd.read_csv(
            self.root_dir / "images.txt",
            sep=" ",
            names=["image_id", "filepath"],
        )

        labels = pd.read_csv(
            self.root_dir / "image_class_labels.txt",
            sep=" ",
            names=["image_id", "class_id"],
        )

        splits = pd.read_csv(
            self.root_dir / "train_test_split.txt",
            sep=" ",
            names=["image_id", "is_train"],
        )

        df = images.merge(labels, on="image_id").merge(splits, on="image_id")

        if use_bbox_crop:
            boxes = pd.read_csv(
                self.root_dir / "bounding_boxes.txt",
                sep=" ",
                names=["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"],
            )
            df = df.merge(boxes, on="image_id")

        if split == "train":
            df = df[df["is_train"] == 1]
        elif split in ["val", "test"]:
            df = df[df["is_train"] == 0]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # CUB labels are 1-200. PyTorch CrossEntropyLoss expects 0-199.
        df["label"] = df["class_id"] - 1

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.images_dir / row["filepath"]
        image = Image.open(image_path).convert("RGB")

        if self.use_bbox_crop:
            image = crop_to_bbox(
                image,
                row["bbox_x"],
                row["bbox_y"],
                row["bbox_w"],
                row["bbox_h"],
            )

        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
