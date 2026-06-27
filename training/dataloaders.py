from torch.utils.data import DataLoader
from torchvision import transforms

from config import IMAGENET_NORMALIZE, TrainConfig
from dataset import CUBDataset


def _build_train_transform(augmentation: str, image_size: int):
    resize_dim = 256 if image_size == 224 else int(image_size * 256 / 224)

    steps = [
        transforms.Resize((resize_dim, resize_dim)),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
    ]

    if augmentation in ("standard", "strong"):
        steps.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        )

    if augmentation == "strong":
        steps.extend([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_NORMALIZE),
    ])

    return transforms.Compose(steps)


def _build_val_transform(image_size: int):
    resize_dim = 256 if image_size == 224 else int(image_size * 256 / 224)

    return transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(**IMAGENET_NORMALIZE),
    ])


def get_dataloaders(cfg: TrainConfig):
    data = cfg.data
    train_transform = _build_train_transform(data.augmentation, data.image_size)
    val_transform = _build_val_transform(data.image_size)

    train_dataset = CUBDataset(
        root_dir=data.data_dir,
        split="train",
        transform=train_transform,
        use_bbox_crop=data.use_bbox_crop,
        val_split_file=data.val_split_file,
    )

    val_dataset = CUBDataset(
        root_dir=data.data_dir,
        split="val",
        transform=val_transform,
        use_bbox_crop=data.use_bbox_crop,
        val_split_file=data.val_split_file,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=data.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=data.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
