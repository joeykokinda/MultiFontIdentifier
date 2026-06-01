import base64
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from detect import analyze_image
from pipeline.classify import _get_model

FONTS_DIR = Path(__file__).parent.parent / "fonts"
_font_db: dict = {}


def load_font_databases():
    for fname in ("tiktok.json", "capcut.json", "canva_popular.json"):
        path = FONTS_DIR / fname
        if not path.exists():
            continue
        source = fname.replace(".json", "")
        with open(path) as f:
            for entry in json.load(f):
                key = entry.get(
                    "google_font", entry.get("google_font_equivalent", entry.get("name", ""))
                ).lower()
                _font_db[key] = {**entry, "source": source}


def enrich_font(font_name: str) -> dict:
    key = font_name.lower()
    for db_key, entry in _font_db.items():
        if key in db_key or db_key in key:
            return entry
    return {}


def run_pipeline(image: Image.Image) -> dict:
    """Same pipeline as the CLI (detect.py), enriched with font-database metadata."""
    return analyze_image(image, enrich=enrich_font)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_font_databases()
    _get_model()
    yield


app = FastAPI(title="MultiFontIdentifier", version="1.0.0", lifespan=lifespan)


@app.post("/detect")
async def detect_fonts(file: Optional[UploadFile] = File(None)):
    if file is None:
        raise HTTPException(400, "No image provided. Send multipart field 'file'.")
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file.")
    return JSONResponse(run_pipeline(image))


class Base64Request(BaseModel):
    image: str


@app.post("/detect/base64")
async def detect_fonts_base64(req: Base64Request):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid base64 image.")
    return JSONResponse(run_pipeline(image))


@app.get("/fonts")
async def list_fonts():
    return JSONResponse({"fonts": list(_font_db.values()), "total": len(_font_db)})


@app.get("/health")
async def health():
    return {"status": "ok"}
