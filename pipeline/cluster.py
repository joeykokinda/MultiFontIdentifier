import numpy as np
from sklearn.cluster import DBSCAN

STYLE_KEYWORDS = ["ExtraBold", "SemiBold", "Bold", "Black", "Light", "Thin", "Medium", "Italic", "Condensed", "Oblique"]


def cluster_by_font(classified_regions: list[dict]) -> list[list[int]]:
    if not classified_regions:
        return []
    if len(classified_regions) == 1:
        return [[0]]

    embeddings = np.array([r["embedding"] for r in classified_regions])
    labels = DBSCAN(eps=0.35, min_samples=1, metric="cosine").fit(embeddings).labels_

    unique_labels = sorted(set(labels))
    return [[i for i, l in enumerate(labels) if l == label] for label in unique_labels]


def aggregate_cluster(classified_regions: list[dict], cluster_indices: list[int]) -> dict:
    font_scores: dict[str, float] = {}

    for idx in cluster_indices:
        for candidate in classified_regions[idx]["top5"]:
            raw = candidate["font"]
            base = raw.split("-")[0].strip() if "-" in raw else raw
            font_scores[base] = font_scores.get(base, 0.0) + candidate["confidence"]

    best_font = max(font_scores, key=font_scores.get)
    total = sum(font_scores.values())
    confidence = font_scores[best_font] / total if total > 0 else 0.0

    top_raw = classified_regions[cluster_indices[0]]["top5"][0]["font"]
    style = "Regular"
    for kw in STYLE_KEYWORDS:
        if kw in top_raw:
            style = kw
            break

    return {
        "font": best_font,
        "style": style,
        "is_italic": "Italic" in top_raw or "Oblique" in top_raw,
        "confidence": round(confidence, 4),
    }
