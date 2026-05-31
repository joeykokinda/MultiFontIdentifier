import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME = os.environ.get("FONT_MODEL", "Storia-AI/font-classify")
CACHE_DIR = str(Path(__file__).parent.parent / "models")

_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        _processor = ViTImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        _model = ViTForImageClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        _model.eval()
    return _model, _processor


def classify_crops(crops: list[Image.Image]) -> list[dict]:
    if not crops:
        return []

    model, processor = _get_model()
    results = []
    batch_size = 16

    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]

        # Hook to extract CLS token embedding from last encoder layer
        embeddings: dict = {}

        def hook(module, input, output):
            embeddings["last"] = output[:, 0, :].detach().cpu().numpy()

        handle = model.vit.encoder.layer[-1].register_forward_hook(hook)

        inputs = processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        handle.remove()

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        top_k = min(5, probs.shape[-1])
        top5 = torch.topk(probs, k=top_k, dim=-1)

        batch_embeddings = embeddings.get("last", np.zeros((len(batch), 768)))

        for j in range(len(batch)):
            top_labels = []
            for k in range(top_k):
                idx = top5.indices[j][k].item()
                conf = top5.values[j][k].item()
                label = model.config.id2label[idx]
                top_labels.append({"font": label, "confidence": round(float(conf), 4)})

            embedding = batch_embeddings[j] if j < len(batch_embeddings) else np.zeros(768)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            results.append({"top5": top_labels, "embedding": embedding})

    return results
