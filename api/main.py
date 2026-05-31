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

from pipeline.classify import classify_crops, _get_model
from pipeline.cluster import aggregate_cluster, cluster_by_font
from pipeline.detect import detect_text_regions
from pipeline.effects import detect_background, detect_outline, detect_text_color
from pipeline.normalize import normalize_crop

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
    regions = detect_text_regions(image)
    if not regions:
        return {"fonts_detected": [], "unknown_regions": []}

    crops = [normalize_crop(image, r["box"], r.get("points")) for r in regions]
    valid = [(r, c) for r, c in zip(regions, crops) if c is not None]
    if not valid:
        return {"fonts_detected": [], "unknown_regions": []}

    valid_regions, valid_crops = zip(*valid)
    classified = classify_crops(list(valid_crops))
    clusters = cluster_by_font(classified)

    fonts_detected = []
    for cluster_indices in clusters:
        agg = aggregate_cluster(classified, cluster_indices)
        db_entry = enrich_font(agg["font"])

        instances = []
        for idx in cluster_indices:
            region = valid_regions[idx]
            box = region["box"]
            x1, y1, x2, y2 = box

            bg = detect_background(image, box)
            outline = detect_outline(image, box)
            text_color = detect_text_color(image, box)

            instances.append({
                "region": box,
                "font_size_px": y2 - y1,
                "text": region["text"],
                "background": bg["background"],
                "background_color": bg["background_color"],
                "text_color": text_color,
                "has_outline": outline["has_outline"],
                "outline_color": outline["outline_color"],
            })

        entry = {
            "font": agg["font"],
            "style": agg["style"],
            "is_italic": agg["is_italic"],
            "confidence": agg["confidence"],
            "source": db_entry.get("source", "google_fonts"),
            "instances": instances,
        }

        nearest = db_entry.get("google_font_equivalent") or db_entry.get("google_font")
        if nearest and nearest.lower() != agg["font"].lower():
            entry["nearest_google_font"] = nearest

        fonts_detected.append(entry)

    return {"fonts_detected": fonts_detected, "unknown_regions": []}


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
