import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image

from pipeline.match import match_fonts
from pipeline.normalize import extract_text_mask

MODEL_REPO = os.environ.get("FONT_MODEL_REPO", "storia/font-classify-onnx")
CACHE_DIR  = str(Path(__file__).parent.parent / "models")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_sess       = None
_classnames = None


def _get_model():
    global _sess, _classnames
    if _sess is None:
        model_path = hf_hub_download(MODEL_REPO, "model.onnx",        cache_dir=CACHE_DIR)
        cfg_path   = hf_hub_download(MODEL_REPO, "model_config.yaml", cache_dir=CACHE_DIR)
        with open(cfg_path) as f:
            _classnames = yaml.safe_load(f)["classnames"]
        _sess = ort.InferenceSession(model_path)
    return _sess, _classnames


def _to_tensor(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None]


def classify_crops(crops: list[Image.Image], regions: list[dict] | None = None) -> list[dict]:
    """
    Two-pass font classification:
      Pass 1 — ViT ONNX model (3,473 Google Fonts)
      Pass 2 — render-and-compare against every downloaded TTF (exact match)

    Pass 2 takes priority when TTFs are present; Pass 1 is the fallback.
    `regions` is the original detect output; used to get ocr text + box size for render-compare.
    """
    if not crops:
        return []

    sess, classnames = _get_model()
    results = []

    for idx, crop in enumerate(crops):
        # --- Pass 1: ViT ONNX ---
        x      = _to_tensor(crop)
        logits = sess.run(None, {"input": x})[0][0]
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        top5   = probs.argsort()[::-1][:5]

        model_top5 = [
            {"font": classnames[i], "confidence": round(float(probs[i]), 4)}
            for i in top5
        ]

        # --- Pass 2: render-and-compare (when TTFs are available) ---
        render_matches: list[dict] = []
        if regions and idx < len(regions):
            region = regions[idx]
            text   = region.get("text", "").strip()
            box    = region.get("box", [0, 0, 100, 50])
            font_h = max(24, box[3] - box[1])

            if text:
                # Get the cleaned mask for comparison (same mask used for ViT)
                orig_image = region.get("_image")   # injected by the caller
                if orig_image is not None:
                    rgb   = np.array(orig_image.crop(box).convert("RGB"))
                    mask  = extract_text_mask(rgb)
                    raw   = match_fonts(text, mask, font_size_px=font_h, top_n=5)
                    render_matches = [
                        {"font": m["font"], "confidence": m["score"],
                         "style": m["style"], "raw": m["raw"]}
                        for m in raw
                    ]

        results.append({
            "model_top5":    model_top5,
            "render_top5":   render_matches,
            # Final top5: render matches first (when available), fallback to model
            "top5": render_matches if render_matches else model_top5,
            "embedding": probs.astype(np.float64),
        })

    return results
