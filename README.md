# Birdbrain

Bird species classifier trained on the [Caltech-UCSD Birds-200-2011 (CUB-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset. Supports EfficientNet-B0 and ResNet50 with YAML-driven staged training.

Planned follow-ups include a web upload interface, inference API, and deployment to `birdbrain.djm-apps.com`.

## Project structure

```
birdbrain/
├── configs/                 # YAML training configs per model/stage
├── splits/                  # val_split.txt (train/val holdout from official train)
├── data/raw/CUB_200_2011/   # Dataset metadata and images/
├── training/
│   ├── config.py            # YAML config loader
│   ├── models.py            # Model builders (EfficientNet, ResNet50)
│   ├── trainer.py           # Training loop, MLflow, checkpoints
│   ├── train.py             # CLI entry point
│   ├── dataset.py           # PyTorch Dataset for CUB
│   ├── dataloaders.py       # Augmentation presets and DataLoaders
│   └── make_labels.py       # Export class index → readable name map
├── models/                  # Saved checkpoints and labels.json (created at runtime)
├── mlflow.db                # MLflow experiment tracking (created at runtime)
├── mlartifacts/             # MLflow run artifacts (created at runtime)
├── db/schema.sql            # Postgres schema for prediction logging (planned)
└── requirements.txt
```

## Requirements

- Python 3.11+
- PyTorch, torchvision, pandas, Pillow (see `requirements.txt`)

A dedicated virtual environment is recommended so global packages (for example, an older `transformers` install) do not trigger PyTorch deprecation warnings when importing torchvision.

## Setup

```bash
cd birdbrain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Download CUB-200-2011 from the [official dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/) and place the contents under `data/raw/CUB_200_2011/`. The directory should include at minimum:

- `images/` — species subfolders with JPEG files
- `images.txt`, `classes.txt`, `image_class_labels.txt`, `train_test_split.txt`

Optional segmentations can live alongside the dataset if you add them later.

**Note:** CUB images are restricted to non-commercial research and educational use. See the dataset website for terms.

## Training

Training is driven by YAML configs in `configs/`. Run from the `training/` directory:

```bash
cd training
python train.py --config ../configs/efficientnet_stage1_head.yaml
```

### EfficientNet staged pipeline (historical)

Each stage loads the prior checkpoint and saves a new one under `models/`:

```bash
python train.py --config ../configs/efficientnet_stage1_head.yaml
python train.py --config ../configs/efficientnet_stage2_last_block.yaml
python train.py --config ../configs/efficientnet_stage3_finetune.yaml
python train.py --config ../configs/efficientnet_stage4_finetune.yaml
python train.py --config ../configs/efficientnet_stage5_bbox.yaml   # ~77% val acc
```

The `recorded:` section in each YAML documents observed results from completed runs.

### ResNet50 staged pipeline

```bash
python train.py --config ../configs/resnet50_stage1_head.yaml
python train.py --config ../configs/resnet50_stage2_finetune.yaml
python train.py --config ../configs/resnet50_stage3_bbox.yaml
```

Training uses the official CUB train/test split (~5,994 train / ~5,794 test). Checkpoints include model weights, normalization settings, and validation accuracy for use at inference time.

### Generate label map

Export a JSON map from class index (0–199) to a readable species name:

```bash
cd training
python make_labels.py
```

Output: `models/labels.json`

## Experiment tracking (MLflow)

Training runs are logged to a local SQLite database at `mlflow.db` (artifacts in `mlartifacts/`).

Start the UI from the project root:

```bash
cd birdbrain
source .venv/bin/activate
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"
```

Open [http://localhost:5000](http://localhost:5000) to compare runs, metrics, and saved checkpoints.

## Model output (planned inference format)

```json
{
  "prediction": "Cardinal",
  "confidence": 0.94,
  "top_5": [
    ["Cardinal", 0.94],
    ["Scarlet Tanager", 0.03],
    ["Summer Tanager", 0.02]
  ]
}
```

Class names in training follow CUB naming (for example, `Cardinal` rather than `Northern Cardinal`). The label map script converts folder-style names to spaced strings for display.

## Dataset citation

This project uses the [Caltech-UCSD Birds-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset. If you use it in published work, please cite:

> **The Caltech-UCSD Birds-200-2011 Dataset**  
> Wah, C.; Branson, S.; Welinder, P.; Perona, P.; Belongie, S.  
> California Institute of Technology, 2011.  
> Technical Report CNS-TR-2011-001
