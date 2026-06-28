"""Birdbrain inference API."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.registry import ModelCache, get_settings
from api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    cache = ModelCache()
    cache.configure(
        models_config=settings["models_config"],
        labels_path=settings["labels_path"],
        data_dir=settings["data_dir"],
    )
    app.state.settings = settings
    app.state.model_cache = cache
    logger.info(
        "API ready on device=%s with models: %s",
        cache.device,
        list(cache.loaded.keys()),
    )
    yield


app = FastAPI(title="Birdbrain API", lifespan=lifespan)
app.include_router(router)
