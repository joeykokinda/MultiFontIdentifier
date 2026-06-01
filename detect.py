#!/usr/bin/env python3
"""
CLI: python detect.py images/slide.png
Detects all fonts in an image and prints JSON to stdout.
"""
import json
import sys
from pathlib import Path

from PIL import Image

from pipeline.attributes import detect_decoration, fit_weight, text_case, width_class
from pipeline.classify import classify_crops
from pipeline.cluster import aggregate_cluster, cluster_by_font
from pipeline.detect import detect_text_regions
from pipeline.effects import detect_background, detect_outline, detect_text_color
from pipeline.layout import analyze_block, analyze_line, group_blocks
from pipeline.normalize import normalize_crop
from pipeline.taxonomy import classify_detection
from pipeline.verify import font_verdict, overall_verdict


def run(image_path: str) -> dict:
    return analyze_image(Image.open(image_path).convert("RGB"))


def analyze_image(image: Image.Image, enrich=None) -> dict:
    """
    Core pipeline shared by the CLI and the API. `enrich(font_name) -> dict`
    is an optional hook (used by the API) to merge font-database metadata
    such as `source` / `nearest_google_font` into each detected font.
    """
    regions = detect_text_regions(image)
    if not regions:
        return {"fonts_detected": [], "text_blocks": [], "verification": {}}

    crops = [normalize_crop(image, r["box"], r.get("points")) for r in regions]
    valid = [(r, c) for r, c in zip(regions, crops) if c is not None]
    if not valid:
        return {"fonts_detected": [], "text_blocks": [], "verification": {}}

    valid_regions, valid_crops = zip(*valid)
    # Inject the source image so Pass 2 (render-and-compare) can re-crop each line.
    regions_for_classify = [{**r, "_image": image} for r in valid_regions]
    classified = classify_crops(list(valid_crops), regions_for_classify)
    clusters = cluster_by_font(classified)

    image_w, image_h = image.size
    fonts_detected = []
    all_line_verdicts = []
    for cluster_indices in clusters:
        agg = aggregate_cluster(classified, cluster_indices)
        raw, style = agg.get("raw"), agg["style"]
        instances = []
        rep = None   # representative (top5, verdict) for family classification
        for idx in cluster_indices:
            region = valid_regions[idx]
            box = region["box"]
            x1, y1, x2, y2 = box
            text = region["text"]
            bg = detect_background(image, box)
            outline = detect_outline(image, box)
            line_layout = analyze_line(image, box, text, raw, style, region.get("points"))
            size = line_layout.get("font_size_px", y2 - y1)
            top5 = classified[idx].get("top5", [])
            verdict = font_verdict(top5)
            all_line_verdicts.append(verdict)
            if rep is None or verdict["match_score"] > rep[1]["match_score"]:
                rep = (top5, verdict)
            instances.append({
                "region": box,
                "text": text,
                "case": text_case(text),
                # size & spacing
                "font_size_px": size,
                "text_ink_height_px": line_layout.get("text_ink_height_px"),
                "letter_spacing_px": line_layout.get("letter_spacing_px"),
                "letter_spacing_em": line_layout.get("letter_spacing_em"),
                "word_spacing_px": line_layout.get("word_spacing_px"),
                "word_spacing_em": line_layout.get("word_spacing_em"),
                # weight / width
                **fit_weight(style, raw),
                **width_class(raw),
                # orientation
                "rotation_deg": line_layout.get("rotation_deg"),
                "slant_deg": line_layout.get("slant_deg"),
                # color / decoration
                "text_color": detect_text_color(image, box),
                "background": bg["background"],
                "background_color": bg["background_color"],
                "has_outline": outline["has_outline"],
                "outline_color": outline["outline_color"],
                **detect_decoration(image, box),
                # honesty
                "font_verification": verdict,
            })
        entry = {
            "font": agg["font"],
            "style": agg["style"],
            "is_italic": agg["is_italic"],
            "confidence": agg["confidence"],
            "classification": classify_detection(*rep) if rep else {},
            "layout": analyze_block(instances, image_w, image_h),
            "instances": instances,
        }
        if enrich:
            db = enrich(agg["font"]) or {}
            if db.get("source"):
                entry["source"] = db["source"]
            nearest = db.get("google_font_equivalent") or db.get("google_font")
            if nearest and nearest.lower() != agg["font"].lower():
                entry["nearest_google_font"] = nearest
        fonts_detected.append(entry)

    text_blocks = group_blocks(list(valid_regions), image_w, image_h)
    return {
        "fonts_detected": fonts_detected,
        "text_blocks": text_blocks,
        "verification": overall_verdict(all_line_verdicts),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path>", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(run(sys.argv[1]), indent=2))
