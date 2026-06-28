# Config reference

Training is configured via YAML files in [`configs/`](../configs/). Loaded by [`training/config.py`](../training/config.py).

## Example

```yaml
experiment_name: bird_classifier_resnet50_stage5
run_name: resnet50-stage5-bbox-crop
phase: "5"

model:
  name: resnet50
  num_classes: 200
  pretrained: true
  checkpoint_path: models/birdbrain_resnet50_v1-3.pt
  output_path: models/birdbrain_resnet50_v1-4.pt

training:
  epochs: 20
  batch_size: 32
  optimizer: adamw
  learning_rate: 0.0001
  backbone_learning_rate: 0.00005
  weight_decay: 0.0001
  label_smoothing: 0.1
  freeze_backbone: true
  unfreeze_blocks: 4
  scheduler:
    patience: 2
    factor: 0.5
  early_stopping_patience: 5

data:
  data_dir: data/raw/CUB_200_2011
  image_size: 224
  use_bbox_crop: true
  augmentation: strong
  num_workers: 4
  val_split_file: splits/val_split.txt

mlflow:
  experiment: birdbrain-cub200

recorded:
  notes: "Manual notes only — not auto-logged"
  best_val_acc: 0.72   # optional historical note
```

## Top-level fields

| Field | Required | Description |
|-------|----------|-------------|
| `experiment_name` | no | Logical experiment name (defaults to run_name) |
| `run_name` | no | MLflow run name (defaults to config filename) |
| `phase` | no | Stage number string; logged as MLflow tag |
| `recorded` | no | Free-form notes for git; **not** sent to MLflow |

## `model`

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | `efficientnet_b0` or `resnet50` |
| `num_classes` | no | Default 200 |
| `pretrained` | no | Default `true` — ImageNet weights |
| `checkpoint_path` | no | Prior stage `.pt`; `null` for stage 1 |
| `output_path` | yes | Where to save best-val checkpoint |

## `training`

| Field | Required | Description |
|-------|----------|-------------|
| `epochs` | no | Default 10 |
| `batch_size` | no | Default 32 |
| `optimizer` | no | `adam`, `adamw`, or `sgd` |
| `learning_rate` | no | Head LR (or sole LR if no split) |
| `backbone_learning_rate` | no | Backbone LR when `unfreeze_blocks > 0` |
| `weight_decay` | no | Default 0 |
| `label_smoothing` | no | CrossEntropyLoss smoothing; 0.1 in stage 5 |
| `freeze_backbone` | no | Always `true` in practice; freeze policy uses `unfreeze_blocks` |
| `unfreeze_blocks` | no | See [Models](models.md) |
| `scheduler.patience` | no | ReduceLROnPlateau patience on val acc |
| `scheduler.factor` | no | LR multiply factor on plateau |
| `early_stopping_patience` | no | Stop after N epochs without val gain; `null` = disabled |

## `data`

| Field | Required | Description |
|-------|----------|-------------|
| `data_dir` | no | Default `data/raw/CUB_200_2011` |
| `image_size` | no | Default 224 |
| `use_bbox_crop` | no | Crop to CUB bounding box before transform |
| `augmentation` | no | `minimal`, `standard`, or `strong` |
| `num_workers` | no | DataLoader workers; default 4 |
| `val_split_file` | no | Default `splits/val_split.txt` |

## `mlflow`

| Field | Required | Description |
|-------|----------|-------------|
| `experiment` | no | Default `birdbrain-cub200` |
| `tracking_uri` | no | Default `sqlite:///<project_root>/mlflow.db` |

## Stage progression defaults

Values that typically change across stages:

| Stage | unfreeze | aug | bbox | LR pattern | early stop |
|-------|----------|-----|------|------------|------------|
| 1 | 0 | minimal | no | single LR 1e-3 | off |
| 2 | 1 | minimal | no | single LR 1e-3 | off |
| 3 | 2–3 | standard | no | split 1e-4 / 5e-4 | off |
| 4 | 2–3 | standard | no | same as 3 | off |
| 5 | all | strong | yes | split 5e-5 / 1e-4 + smoothing | patience 5 |

ResNet stage 5 uses `unfreeze_blocks: 4` (all layers). EfficientNet stage 5 uses `5`.

## Allowed values (enforced at load time)

```python
ALLOWED_MODELS = {"efficientnet_b0", "resnet50"}
ALLOWED_OPTIMIZERS = {"adam", "adamw", "sgd"}
ALLOWED_AUGMENTATIONS = {"minimal", "standard", "strong"}
```

Invalid values raise `ValueError` when loading config.
