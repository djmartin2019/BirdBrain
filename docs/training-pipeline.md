# Training pipeline

## Entry point

```bash
cd training
python train.py --config ../configs/<model>_stage<N>_<name>.yaml
```

[`train.py`](../training/train.py) loads YAML via [`config.py`](../training/config.py) and calls [`trainer.train()`](../training/trainer.py).

## Staged fine-tuning strategy

Both EfficientNet-B0 and ResNet50 follow five stages. Each stage:

1. Loads the previous stage's checkpoint (except stage 1)
2. Applies a freeze/unfreeze policy
3. Trains on **train** split, validates on **val** split
4. Saves the best-val checkpoint to `model.output_path`
5. Logs metrics to MLflow

### EfficientNet-B0 stages

| Stage | Config | Unfreeze | Aug | Bbox | Output |
|-------|--------|----------|-----|------|--------|
| 1 | `efficientnet_stage1_head.yaml` | 0 blocks (head only) | minimal | no | `birdbrain_v1.pt` |
| 2 | `efficientnet_stage2_last_block.yaml` | 1 block | minimal | no | `birdbrain_v1-1.pt` |
| 3 | `efficientnet_stage3_finetune.yaml` | 3 blocks | standard | no | `birdbrain_v1-2.pt` |
| 4 | `efficientnet_stage4_finetune.yaml` | 3 blocks | standard | no | `birdbrain_v1-3.pt` |
| 5 | `efficientnet_stage5_bbox.yaml` | 5 blocks | strong | yes | `birdbrain_v1-4.pt` |

### EfficientNet-B0 iNat birds (4 stages)

Trains on iNat 2021 birds only (`num_classes: 1486`). Stage 1 loads the CUB stage-5 checkpoint as backbone init (`birdbrain_v1-4.pt`); no bbox stage.

| Stage | Config | Unfreeze | Aug | Output |
|-------|--------|----------|-----|--------|
| 1 | `efficientnet_inat_stage1_head.yaml` | 0 blocks (head only) | minimal | `birdbrain_inat_v1.pt` |
| 2 | `efficientnet_inat_stage2_last_block.yaml` | 1 block | minimal | `birdbrain_inat_v1-1.pt` |
| 3 | `efficientnet_inat_stage3_finetune.yaml` | 3 blocks | standard | `birdbrain_inat_v1-2.pt` |
| 4 | `efficientnet_inat_stage4_finetune.yaml` | 5 blocks | strong | `birdbrain_inat_v1-3.pt` |

Run after verifying data: `python scripts/verify_inat2021.py`. Final test eval uses `--split test` (official iNat `val.json`, birds filtered).

### ResNet50 stages

| Stage | Config | Unfreeze | Aug | Bbox | Output |
|-------|--------|----------|-----|------|--------|
| 1 | `resnet50_stage1_head.yaml` | fc only | minimal | no | `birdbrain_resnet50_v1.pt` |
| 2 | `resnet50_stage2_last_block.yaml` | layer4 | minimal | no | `birdbrain_resnet50_v1-1.pt` |
| 3 | `resnet50_stage3_finetune.yaml` | layer3–4 | standard | no | `birdbrain_resnet50_v1-2.pt` |
| 4 | `resnet50_stage4_finetune.yaml` | layer2–4 | standard | no | `birdbrain_resnet50_v1-3.pt` |
| 5 | `resnet50_stage5_bbox.yaml` | layer1–4 | strong | yes | `birdbrain_resnet50_v1-4.pt` |

Run stages **in order**. Stage 5 is the production candidate for each model family.

## Training loop (`trainer.py`)

Per epoch:

1. **Train** — forward/backward on train loader; running loss and accuracy
2. **Validate** — eval mode on val loader; no gradients
3. **LR schedule** — `ReduceLROnPlateau` on val accuracy (max mode, patience/factor from config)
4. **Checkpoint** — if val acc improves, save weights + metadata to `output_path`
5. **MLflow** — log per-epoch metrics and checkpoint artifact on improvement
6. **Early stopping** — if configured, stop after N epochs without val improvement (stage 5)

### Optimizers

- EfficientNet configs: `adam`
- ResNet50 configs: `adamw`

When `unfreeze_blocks > 0` and `backbone_learning_rate` is set, the optimizer uses **split learning rates**: lower LR for unfrozen backbone, higher LR for head.

### Console output example

```
Split sizes — train: 5394, val: 600 (official test held out for final eval)
Epoch 1/10
Train Loss: 4.6134 | Train Acc: 0.1161
Val   Loss: 3.8201 | Val   Acc: 0.2750
Gap: -0.1589 | LRs: ['1.00e-03']
Saved new best model: .../models/birdbrain_resnet50_v1.pt
```

## MLflow tracking

Each `train.py` run creates one MLflow run in experiment `birdbrain-cub200` (configurable).

**Stored locally:**

- `mlflow.db` — SQLite tracking store
- `mlartifacts/` — checkpoint copies and artifacts

**Logged per run:**

| Type | Examples |
|------|----------|
| Tags | `model`, `phase` |
| Params | hyperparams, config path, split sizes, device |
| Metrics (per epoch) | `train_loss`, `train_acc`, `val_loss`, `val_acc`, `train_val_gap`, `lr_group_*` |
| Metrics (summary) | `final_best_val_acc`, `training_minutes`, `prior_val_acc` |
| Artifacts | best checkpoint on each val improvement |

**View runs:**

```bash
cd birdbrain
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"
```

Open http://localhost:5000. Filter by run name (e.g. `resnet50-stage3-finetune`) or tags.

**Note:** The `recorded:` section in YAML files is manual documentation only — it is **not** written to MLflow automatically.

## Path resolution

Paths in YAML are relative to the **project root** (`birdbrain/`). CLI tools also accept paths relative to the current working directory when run from `training/` (see `resolve_cli_path` in `config.py`).

## Split discipline checklist

- [x] Training uses train + val only
- [x] Best checkpoint selected on val
- [ ] Official test evaluated **once** after final stage via `evaluate.py`
- [ ] Do not tune hyperparameters based on test results and re-test
