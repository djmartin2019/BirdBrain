# Birdbrain API

FastAPI inference service for the bird classifier. Loads production checkpoints at startup and serves top-5 predictions with percentage scores.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status, device, loaded model IDs |
| GET | `/api/models` | Models available for selection |
| POST | `/api/predict` | Upload image → top-5 prediction JSON |

### `GET /api/models`

```json
{
  "models": [
    {
      "id": "efficientnet_b0",
      "name": "EfficientNet-B0",
      "description": "Lightweight backbone; ~77% val on CUB",
      "default": true
    }
  ]
}
```

### `POST /api/predict`

Multipart form fields:

- `file` — JPEG, PNG, or WebP (max 10 MB by default)
- `model_id` — e.g. `efficientnet_b0` or `resnet50`

```json
{
  "model_id": "efficientnet_b0",
  "model_name": "EfficientNet-B0",
  "prediction": "Cardinal",
  "confidence": 94.2,
  "top_5": [
    { "species": "Cardinal", "percent": 94.2 },
    { "species": "Scarlet Tanager", "percent": 3.1 }
  ]
}
```

Percentages are softmax probabilities × 100 (sum ≈ 100).

## Model registry

Configured in [`models.yaml`](models.yaml). Checkpoints are loaded eagerly at startup. If a checkpoint file is missing, that model is omitted from `/api/models` (the service still starts).

| `id` | Checkpoint |
|------|------------|
| `efficientnet_b0` | `models/birdbrain_v1-4.pt` |
| `resnet50` | `models/birdbrain_resnet50_v1-4.pt` |

## Upload preprocessing note

Stage-5 checkpoints were trained with CUB bounding-box crops (`use_bbox_crop: true`). User uploads have no bbox metadata, so inference uses full-image eval transforms (resize → center crop → normalize). Accuracy on arbitrary photos may differ from CUB test metrics.

Shared logic lives in [`training/inference.py`](../training/inference.py).

## Setup

From the repo root:

```bash
pip install -r requirements.txt -r api/requirements.txt
```

Copy `.env.example` to `.env` only if you need non-default paths.

## Run

From the repo root:

```bash
uvicorn api.main:app --reload --port 8000
```

## Development with frontend

The SvelteKit dev server proxies `/api` to `http://localhost:8000` (see `web/vite.config.ts`).

Terminal 1:

```bash
uvicorn api.main:app --reload --port 8000
```

Terminal 2:

```bash
cd web && npm run dev
```

Open http://localhost:5173

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BIRDBRAIN_MODELS_CONFIG` | `api/models.yaml` | Model registry |
| `BIRDBRAIN_LABELS_PATH` | `models/labels.json` | Class name map |
| `BIRDBRAIN_DATA_DIR` | `data/raw/CUB_200_2011` | CUB path for `classes.txt` fallback |
| `BIRDBRAIN_MAX_UPLOAD_MB` | `10` | Max upload size |
