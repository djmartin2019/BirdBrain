from pathlib import Path
from PIL import Image

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CUBDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.transform = transform

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
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
