"""Shared inference utilities for API and offline scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from config import IMAGENET_NORMALIZE, resolve_cli_path
from dataloaders import _build_val_transform
from models import build_model_for_inference


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    model_name: str
    num_classes: int
    image_size: int
    use_bbox_crop: bool
    normalization: dict[str, list[float]]
    best_val_acc: float | None


@dataclass
class TopKPrediction:
    species: str
    percent: float


def load_class_names(
    labels_path: Path | None,
    data_dir: Path,
    num_classes: int,
) -> dict[int, str]:
    """Load index -> class name map from labels.json or CUB classes.txt."""
    if labels_path is not None:
        labels_path = resolve_cli_path(labels_path)
        if labels_path.exists() and labels_path.stat().st_size > 0:
            with open(labels_path) as f:
                raw = json.load(f)
            if raw:
                return {int(k): v for k, v in raw.items()}

    classes_file = data_dir / "classes.txt"
    if not classes_file.exists():
        return {i: str(i) for i in range(num_classes)}

    classes = pd.read_csv(classes_file, sep=" ", names=["class_id", "class_name"])
    return {
        int(row.class_id) - 1: row.class_name.split(".", 1)[1].replace("_", " ")
        for row in classes.itertuples()
    }


def preprocess_image(
    image: Image.Image,
    image_size: int,
    normalization: dict[str, list[float]] | None = None,
) -> torch.Tensor:
    """Apply eval-time transforms to a PIL image; returns a batch tensor [1, C, H, W]."""
    norm = normalization or IMAGENET_NORMALIZE
    transform = _build_val_transform(image_size)
    tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> LoadedCheckpoint:
    """Load model weights and metadata from a .pt checkpoint."""
    if device is None:
        device = get_device()

    checkpoint_path = resolve_cli_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]

    model = build_model_for_inference(model_name, num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    return LoadedCheckpoint(
        model=model,
        model_name=model_name,
        num_classes=num_classes,
        image_size=checkpoint.get("image_size", 224),
        use_bbox_crop=checkpoint.get("use_bbox_crop", False),
        normalization=checkpoint.get("normalization", IMAGENET_NORMALIZE),
        best_val_acc=checkpoint.get("best_val_acc"),
    )


def predict_topk(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    class_names: dict[int, str],
    device: torch.device | None = None,
    k: int = 5,
) -> list[TopKPrediction]:
    """Run forward pass and return top-k species with softmax percentages."""
    if device is None:
        device = get_device()

    model.eval()
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        topk = min(k, probs.size(0))
        values, indices = probs.topk(topk)

    results: list[TopKPrediction] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        species = class_names.get(index, str(index))
        results.append(TopKPrediction(species=species, percent=round(value * 100, 1)))

    return results
