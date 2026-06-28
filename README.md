# Birdbrain

Bird species classifier trained on the [Caltech-UCSD Birds-200-2011 (CUB-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset. Supports EfficientNet-B0 and ResNet50 with YAML-driven staged training.

**Documentation:** See [`docs/`](docs/README.md) for pipeline architecture, data splits, models, evaluation, and config reference.

Planned follow-ups include deployment to `birdbrain.djm-apps.com` (SvelteKit frontend in `web/`, FastAPI in `api/`).

## Project structure

```
birdbrain/
├── configs/                 # YAML training configs per model/stage
├── docs/                    # Technical documentation (pipeline, data, models, eval)
├── web/                     # SvelteKit frontend (Identify, About, Docs, Citation)
├── api/                     # FastAPI inference service (+ Docker image)
├── docker/                  # nginx config and web production Dockerfile
├── docker-compose.yml       # Run web + API (port 3012 by default)
├── splits/                  # val_split.txt (train/val holdout from official train)
├── data/raw/CUB_200_2011/   # Dataset metadata and images/
├── training/
│   ├── config.py            # YAML config loader
│   ├── models.py            # Model builders (EfficientNet, ResNet50)
│   ├── trainer.py           # Training loop, MLflow, checkpoints
│   ├── train.py             # CLI entry point
│   ├── dataset.py           # PyTorch Dataset for CUB
│   ├── dataloaders.py       # Augmentation presets and DataLoaders
│   ├── evaluate.py          # Checkpoint eval on val/test
│   ├── confusion_matrix.py  # Per-class analysis and Plotly charts
│   └── make_labels.py       # Export class index → readable name map
├── models/                  # Training checkpoints and analysis output (gitignored)
├── prod-models/             # Production inference checkpoints only (weights gitignored)
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

### Train / val / test splits

CUB ships an official **train** (~5,994) and **test** (~5,794) split. For controlled experiments we further split official train into:

| Split | Source | Used for |
|-------|--------|----------|
| **train** | 90% of official train | Weight updates |
| **val** | 10% of official train | Early stopping, checkpoint selection |
| **test** | Official CUB test | Final evaluation only (not used during training) |

The val assignment lives in [`splits/val_split.txt`](splits/val_split.txt) (committed to the repo). Regenerate with:

```bash
python scripts/create_val_split.py
```

Configure path in YAML via `data.val_split_file` (default: `splits/val_split.txt`).

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

Mirrors the five EfficientNet stages (head → last block → progressive fine-tune → bbox crop). Checkpoints chain under `models/birdbrain_resnet50_v1*.pt`:

| Stage | Config | Unfrozen | Aug | Bbox |
|-------|--------|----------|-----|------|
| 1 | `resnet50_stage1_head.yaml` | fc only | minimal | no |
| 2 | `resnet50_stage2_last_block.yaml` | layer4 | minimal | no |
| 3 | `resnet50_stage3_finetune.yaml` | layer3–4 | standard | no |
| 4 | `resnet50_stage4_finetune.yaml` | layer2–4 | standard | no |
| 5 | `resnet50_stage5_bbox.yaml` | layer1–4 | strong | yes |

```bash
python train.py --config ../configs/resnet50_stage1_head.yaml
python train.py --config ../configs/resnet50_stage2_last_block.yaml
python train.py --config ../configs/resnet50_stage3_finetune.yaml
python train.py --config ../configs/resnet50_stage4_finetune.yaml
python train.py --config ../configs/resnet50_stage5_bbox.yaml
```

After stage 5, run final test eval once:

```bash
python evaluate.py --config ../configs/resnet50_stage5_bbox.yaml \
  --checkpoint ../models/birdbrain_resnet50_v1-4.pt --split test --mlflow
```

Checkpoints include model weights, normalization settings, and validation accuracy for use at inference time.

### Generate label map

Export a JSON map from class index (0–199) to a readable species name:

```bash
cd training
python make_labels.py
```

Output: `models/labels.json`

### Final evaluation (official test set)

After training, evaluate a checkpoint once on the held-out **test** split:

```bash
cd training
python evaluate.py --config ../configs/efficientnet_stage5_bbox.yaml \
  --checkpoint ../models/birdbrain_v1-4.pt --split test
```

Use `--split val` to re-score on the validation holdout. Add `--mlflow` to log the eval run. Writes a JSON report next to the checkpoint (e.g. `birdbrain_v1-4.eval.json`).

### Confusion matrix analysis

Standalone script for per-class errors — worst-recall bar chart, top-K submatrix heatmap, and confused pairs CSV. Run after training or evaluation:

```bash
cd training
python confusion_matrix.py --config ../configs/resnet50_stage5_bbox.yaml \
  --checkpoint ../models/birdbrain_resnet50_v1-4.pt --split test --top-k 20
```

Writes to `<checkpoint_stem>_confusion/` by default:

- `worst_recall_bar.html` — open in a browser
- `topk_submatrix.html` — K×K heatmap for worst classes
- `confused_pairs.csv` — most common misclassification pairs
- `per_class_recall.csv`, `confusion_matrix.csv`, `summary.json`

Add `--normalize-rows` for a row-normalized submatrix heatmap.

## Web frontend

SvelteKit app in [`web/`](web/README.md) — upload UI with model picker, about, docs, and citation pages.

Terminal 1 — API (from repo root):

```bash
pip install -r requirements.txt -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

Terminal 2 — frontend:

```bash
cd web
npm install
npm run dev
```

Open http://localhost:5173 (Vite proxies `/api` to the API on port 8000).

See [`api/README.md`](api/README.md) for endpoints and configuration.

## Docker (production)

Run the static frontend and inference API together with nginx on port **3012**:

```bash
# Ensure production checkpoints exist in prod-models/ (see prod-models/README.md)
docker compose up --build
```

Open http://localhost:3012

The `web` service serves the SvelteKit build and proxies `/api` to the API container. Production checkpoints are mounted read-only from `./prod-models`. Class names are baked into the API image via [`api/labels.json`](api/labels.json).

After retraining, promote checkpoints with `./scripts/sync-prod-models.sh` (copies final `.pt` files from `models/`).

Override the host port:

```bash
BIRDBRAIN_PORT=3012 docker compose up --build
```

Stop and remove containers:

```bash
docker compose down
```

### Deployment scripts (VPS)

From the repo root on the server:

```bash
chmod +x scripts/deploy-full.sh scripts/deploy-web.sh   # once
```

| Script | When to use |
|--------|-------------|
| `./scripts/deploy-full.sh` | First deploy, API changes, dependency updates |
| `./scripts/deploy-full.sh --no-cache` | Same, but ignore Docker cache (slow) |
| `./scripts/deploy-web.sh` | Frontend-only changes (pages, styles, copy) |

Both scripts run `git pull --ff-only`, then the appropriate `docker compose` build/up commands. Copy [`.env.example`](.env.example) to `.env` on the server to set `BIRDBRAIN_PORT=3012`.

Ship `prod-models/*.pt` to the VPS separately (rsync); weights are not in git. Use `./scripts/sync-prod-models.sh` locally after promoting new checkpoints from `models/`.

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

See [docs/citation.md](docs/citation.md) for CUB-200-2011 and iNaturalist 2021 citations, BibTeX, and research/educational use terms.
