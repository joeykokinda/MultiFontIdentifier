#!/usr/bin/env python3
"""
CLI: python detect.py images/slide.png
Detects all fonts in an image and prints JSON to stdout.
"""
import json
import sys
from pathlib import Path

from PIL import Image

from pipeline.classify import classify_crops
from pipeline.cluster import aggregate_cluster, cluster_by_font
from pipeline.detect import detect_text_regions
from pipeline.effects import detect_background, detect_outline, detect_text_color
from pipeline.normalize import normalize_crop


def run(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")

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
        instances = []
        for idx in cluster_indices:
            region = valid_regions[idx]
            box = region["box"]
            x1, y1, x2, y2 = box
            bg = detect_background(image, box)
            outline = detect_outline(image, box)
            instances.append({
                "region": box,
                "font_size_px": y2 - y1,
                "text": region["text"],
                "background": bg["background"],
                "background_color": bg["background_color"],
                "text_color": detect_text_color(image, box),
                "has_outline": outline["has_outline"],
                "outline_color": outline["outline_color"],
            })
        fonts_detected.append({
            "font": agg["font"],
            "style": agg["style"],
            "is_italic": agg["is_italic"],
            "confidence": agg["confidence"],
            "instances": instances,
        })

    return {"fonts_detected": fonts_detected}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path>", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(run(sys.argv[1]), indent=2))
