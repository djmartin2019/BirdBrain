"""FastAPI routes for bird classification inference."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = PROJECT_ROOT / "training"
for path in (PROJECT_ROOT, TRAINING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from api.registry import ModelCache
from inference import predict_topk, preprocess_image

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


def _get_cache(request: Request) -> ModelCache:
    return request.app.state.model_cache


@router.get("/health")
def health(request: Request):
    cache = _get_cache(request)
    return {
        "status": "ok",
        "device": str(cache.device),
        "models_loaded": list(cache.loaded.keys()),
    }


@router.get("/api/models")
def list_models(request: Request):
    cache = _get_cache(request)
    models = []
    default_set = False

    for entry in cache.available_models():
        is_default = entry.default
        if is_default:
            default_set = True
        item = {
            "id": entry.id,
            "name": entry.name,
            "description": entry.description,
        }
        if entry.best_val_acc is not None:
            item["best_val_acc"] = entry.best_val_acc
        if is_default:
            item["default"] = True
        models.append(item)

    if models and not default_set:
        models[0]["default"] = True

    return {"models": models}


@router.post("/api/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_id: str = Form(...),
):
    cache = _get_cache(request)
    settings = request.app.state.settings
    max_bytes = settings["max_upload_mb"] * 1024 * 1024

    loaded = cache.get(model_id)
    if loaded is None:
        if model_id not in {e.id for e in cache.entries}:
            raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
        raise HTTPException(
            status_code=503,
            detail=f"Model {model_id!r} is not loaded (checkpoint missing or failed to load)",
        )

    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type or 'unknown'}. Use JPEG, PNG, or WebP.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large (max {settings['max_upload_mb']} MB)",
        )

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    ckpt = loaded.checkpoint
    tensor = preprocess_image(image, ckpt.image_size, ckpt.normalization)
    top_k = predict_topk(
        ckpt.model,
        tensor,
        loaded.class_names,
        device=cache.device,
        k=5,
    )

    if not top_k:
        raise HTTPException(status_code=500, detail="Model returned no predictions")

    top = top_k[0]
    return {
        "model_id": loaded.entry.id,
        "model_name": loaded.entry.name,
        "prediction": top.species,
        "confidence": top.percent,
        "top_5": [
            {"species": item.species, "percent": item.percent}
            for item in top_k
        ],
    }
