from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_MODELS = {"efficientnet_b0", "resnet50"}
ALLOWED_OPTIMIZERS = {"adam", "adamw", "sgd"}
ALLOWED_AUGMENTATIONS = {"minimal", "standard", "strong"}
ALLOWED_DATASETS = {"cub", "inat2021"}

IMAGENET_NORMALIZE = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


@dataclass
class ModelConfig:
    name: str
    num_classes: int
    pretrained: bool
    checkpoint_path: Path | None
    output_path: Path


@dataclass
class SchedulerConfig:
    patience: int = 2
    factor: float = 0.5


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    backbone_learning_rate: float | None
    weight_decay: float
    label_smoothing: float
    freeze_backbone: bool
    unfreeze_blocks: int
    scheduler: SchedulerConfig
    early_stopping_patience: int | None


@dataclass
class DataConfig:
    dataset: str
    data_dir: Path
    image_size: int
    use_bbox_crop: bool
    augmentation: str
    num_workers: int
    val_split_file: Path


@dataclass
class MlflowConfig:
    experiment: str
    tracking_uri: str | None = None


@dataclass
class TrainConfig:
    experiment_name: str
    run_name: str
    phase: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    mlflow: MlflowConfig
    config_path: Path
    recorded: dict[str, Any] = field(default_factory=dict)


def _resolve_path(value: str | None, project_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


def resolve_cli_path(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a CLI path: cwd-relative first, then project-root-relative."""
    path = Path(path)
    if path.is_absolute():
        return path.resolve()

    cwd_resolved = (Path.cwd() / path).resolve()
    root_resolved = (project_root / path).resolve()
    if cwd_resolved.exists():
        return cwd_resolved
    if root_resolved.exists():
        return root_resolved
    if ".." in path.parts:
        return cwd_resolved
    return root_resolved


def _require_section(raw: dict, key: str) -> dict:
    if key not in raw:
        raise ValueError(f"Missing required config section: {key}")
    return raw[key]


def load_config(config_path: str | Path) -> TrainConfig:
    config_path = Path(config_path).resolve()
    project_root = PROJECT_ROOT

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    model_raw = _require_section(raw, "model")
    training_raw = _require_section(raw, "training")
    data_raw = _require_section(raw, "data")
    mlflow_raw = raw.get("mlflow", {})

    model_name = model_raw.get("name")
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"model.name must be one of {sorted(ALLOWED_MODELS)}, got {model_name!r}"
        )

    optimizer = training_raw.get("optimizer", "adam").lower()
    if optimizer not in ALLOWED_OPTIMIZERS:
        raise ValueError(
            f"training.optimizer must be one of {sorted(ALLOWED_OPTIMIZERS)}, "
            f"got {optimizer!r}"
        )

    augmentation = data_raw.get("augmentation", "minimal").lower()
    if augmentation not in ALLOWED_AUGMENTATIONS:
        raise ValueError(
            f"data.augmentation must be one of {sorted(ALLOWED_AUGMENTATIONS)}, "
            f"got {augmentation!r}"
        )

    dataset = data_raw.get("dataset", "cub").lower()
    if dataset not in ALLOWED_DATASETS:
        raise ValueError(
            f"data.dataset must be one of {sorted(ALLOWED_DATASETS)}, got {dataset!r}"
        )

    use_bbox_crop = bool(data_raw.get("use_bbox_crop", False))
    if dataset == "inat2021" and use_bbox_crop:
        raise ValueError("use_bbox_crop is not supported for dataset 'inat2021'")

    scheduler_raw = training_raw.get("scheduler", {})
    tracking_uri = mlflow_raw.get("tracking_uri")
    if tracking_uri is None:
        tracking_uri = f"sqlite:///{project_root / 'mlflow.db'}"

    if "output_path" not in model_raw:
        raise ValueError("model.output_path is required")

    return TrainConfig(
        experiment_name=raw.get("experiment_name", raw.get("run_name", "training_run")),
        run_name=raw.get("run_name", config_path.stem),
        phase=str(raw.get("phase", "")),
        model=ModelConfig(
            name=model_name,
            num_classes=int(model_raw.get("num_classes", 200)),
            pretrained=bool(model_raw.get("pretrained", True)),
            checkpoint_path=_resolve_path(model_raw.get("checkpoint_path"), project_root),
            output_path=_resolve_path(model_raw["output_path"], project_root),
        ),
        training=TrainingConfig(
            epochs=int(training_raw.get("epochs", 10)),
            batch_size=int(training_raw.get("batch_size", 32)),
            optimizer=optimizer,
            learning_rate=float(training_raw.get("learning_rate", 1e-3)),
            backbone_learning_rate=(
                float(training_raw["backbone_learning_rate"])
                if training_raw.get("backbone_learning_rate") is not None
                else None
            ),
            weight_decay=float(training_raw.get("weight_decay", 0.0)),
            label_smoothing=float(training_raw.get("label_smoothing", 0.0)),
            freeze_backbone=bool(training_raw.get("freeze_backbone", True)),
            unfreeze_blocks=int(training_raw.get("unfreeze_blocks", 0)),
            scheduler=SchedulerConfig(
                patience=int(scheduler_raw.get("patience", 2)),
                factor=float(scheduler_raw.get("factor", 0.5)),
            ),
            early_stopping_patience=(
                int(training_raw["early_stopping_patience"])
                if training_raw.get("early_stopping_patience") is not None
                else None
            ),
        ),
        data=DataConfig(
            dataset=dataset,
            data_dir=_resolve_path(
                data_raw.get("data_dir", "data/raw/CUB_200_2011"),
                project_root,
            ),
            image_size=int(data_raw.get("image_size", 224)),
            use_bbox_crop=use_bbox_crop,
            augmentation=augmentation,
            num_workers=int(data_raw.get("num_workers", 4)),
            val_split_file=_resolve_path(
                data_raw.get(
                    "val_split_file",
                    "splits/inat_val_split.txt"
                    if dataset == "inat2021"
                    else "splits/val_split.txt",
                ),
                project_root,
            ),
        ),
        mlflow=MlflowConfig(
            experiment=mlflow_raw.get("experiment", "birdbrain-cub200"),
            tracking_uri=tracking_uri,
        ),
        config_path=config_path,
        recorded=raw.get("recorded", {}),
    )
