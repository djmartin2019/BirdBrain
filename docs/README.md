# Birdbrain documentation

Technical reference for the CUB-200 bird classifier: data handling, training pipeline, models, evaluation, and result display.

For quick setup and commands, see the [project README](../README.md).

## Contents

| Doc | Description |
|-----|-------------|
| [Overview](overview.md) | End-to-end flow, directory layout, design principles |
| [Data & splits](data-and-splits.md) | CUB dataset, train/val/test splits, bbox crops, augmentations |
| [Training pipeline](training-pipeline.md) | YAML configs, staged fine-tuning, trainer loop, MLflow |
| [Models](models.md) | EfficientNet-B0 & ResNet50, freeze policy, checkpoints |
| [Evaluation & analysis](evaluation-and-analysis.md) | `evaluate.py`, confusion matrix reports, metrics display |
| [Config reference](config-reference.md) | YAML field definitions and stage templates |

## Typical workflow

```
1. Download CUB-200-2011 → data/raw/CUB_200_2011/
2. (Optional) Regenerate val split → splits/val_split.txt
3. Train stages 1–5 via configs/resnet50_stage*.yaml or efficientnet_stage*.yaml
4. Final test eval once → evaluate.py --split test
5. Per-class analysis → confusion_matrix.py --split test
6. View training history → mlflow ui
```

All training and analysis commands run from the `training/` directory unless noted otherwise.
