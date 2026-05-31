import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image

MODEL_REPO = os.environ.get("FONT_MODEL_REPO", "storia/font-classify-onnx")
CACHE_DIR  = str(Path(__file__).parent.parent / "models")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_sess       = None
_classnames = None


def _get_model():
    global _sess, _classnames
    if _sess is None:
        model_path = hf_hub_download(MODEL_REPO, "model.onnx",    cache_dir=CACHE_DIR)
        cfg_path   = hf_hub_download(MODEL_REPO, "model_config.yaml", cache_dir=CACHE_DIR)
        with open(cfg_path) as f:
            _classnames = yaml.safe_load(f)["classnames"]
        _sess = ort.InferenceSession(model_path)
    return _sess, _classnames


def _to_tensor(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None]


def classify_crops(crops: list[Image.Image]) -> list[dict]:
    if not crops:
        return []

    sess, classnames = _get_model()
    results = []

    for crop in crops:
        x      = _to_tensor(crop)
        logits = sess.run(None, {"input": x})[0][0]
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        top5   = probs.argsort()[::-1][:5]

        results.append({
            "top5": [
                {"font": classnames[i], "confidence": round(float(probs[i]), 4)}
                for i in top5
            ],
            # Use prob vector as embedding for clustering (no hook needed for ONNX)
            "embedding": probs.astype(np.float64),
        })

    return results
