"""Model registry and in-memory cache for loaded checkpoints."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = PROJECT_ROOT / "training"

for path in (PROJECT_ROOT, TRAINING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from inference import LoadedCheckpoint, get_device, load_class_names, load_model_from_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    id: str
    name: str
    description: str
    checkpoint: Path
    labels_path: Path | None
    default: bool
    best_val_acc: float | None = None


@dataclass
class LoadedModel:
    entry: ModelEntry
    checkpoint: LoadedCheckpoint
    class_names: dict[int, str]


class ModelCache:
    def __init__(self):
        self.device = get_device()
        self.entries: list[ModelEntry] = []
        self.loaded: dict[str, LoadedModel] = {}

    def configure(
        self,
        models_config: Path,
        labels_path: Path,
        data_dir: Path,
    ):
        self.entries = load_registry(models_config)
        if not self.entries:
            logger.warning("No model entries found in %s", models_config)
            return

        for entry in self.entries:
            if not entry.checkpoint.exists():
                logger.warning(
                    "Skipping model %s: checkpoint not found at %s",
                    entry.id,
                    entry.checkpoint,
                )
                continue

            try:
                checkpoint = load_model_from_checkpoint(entry.checkpoint, self.device)
                if entry.best_val_acc is None and checkpoint.best_val_acc is not None:
                    entry.best_val_acc = checkpoint.best_val_acc

                model_labels_path = entry.labels_path or labels_path
                names = load_class_names(
                    model_labels_path, data_dir, checkpoint.num_classes
                )
                self.loaded[entry.id] = LoadedModel(
                    entry=entry,
                    checkpoint=checkpoint,
                    class_names=names,
                )
                logger.info("Loaded model %s from %s", entry.id, entry.checkpoint)
            except Exception:
                logger.exception("Failed to load model %s", entry.id)

    def available_models(self) -> list[ModelEntry]:
        return [self.loaded[mid].entry for mid in self.loaded]

    def get(self, model_id: str) -> LoadedModel | None:
        return self.loaded.get(model_id)


def load_registry(config_path: Path) -> list[ModelEntry]:
    config_path = _resolve_path(config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    entries: list[ModelEntry] = []
    for item in raw.get("models", []):
        labels = item.get("labels")
        entries.append(
            ModelEntry(
                id=item["id"],
                name=item["name"],
                checkpoint=_resolve_path(item["checkpoint"]),
                labels_path=_resolve_path(labels) if labels else None,
                description=item.get("description", ""),
                default=bool(item.get("default", False)),
                best_val_acc=item.get("best_val_acc"),
            )
        )
    return entries


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_settings() -> dict[str, Path | int]:
    return {
        "models_config": _resolve_path(
            os.environ.get("BIRDBRAIN_MODELS_CONFIG", "api/models.yaml")
        ),
        "labels_path": _resolve_path(
            os.environ.get("BIRDBRAIN_LABELS_PATH", "models/labels.json")
        ),
        "data_dir": _resolve_path(
            os.environ.get("BIRDBRAIN_DATA_DIR", "data/raw/CUB_200_2011")
        ),
        "max_upload_mb": int(os.environ.get("BIRDBRAIN_MAX_UPLOAD_MB", "10")),
    }
