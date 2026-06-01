"""
Render-and-compare font matching.

Takes the OCR text and renders it in every downloaded TTF, then compares
the rendered letterforms against the actual image crop using SSIM.
This gives accurate matches for any font we have a TTF for, including
the full TikTok/IG/CapCut font set.
"""
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TTF_DIR = Path(__file__).parent.parent / "fonts" / "ttf"

# Weight these style suffixes so we prefer the bold/heavy variants
# (creator text is almost always bold)
_STYLE_PRIORITY = {
    "ExtraBold": 1.0,
    "Black":     1.0,
    "Bold":      0.95,
    "SemiBold":  0.85,
    "Medium":    0.7,
    "Regular":   0.6,
    "Light":     0.3,
    "Thin":      0.2,
}


@lru_cache(maxsize=512)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return None


def _available_fonts() -> list[tuple[str, str]]:
    """Return list of (font_name, ttf_path) for all downloaded TTFs."""
    results = []
    for ttf in sorted(TTF_DIR.glob("*.ttf")):
        # Parse family + style from filename: FontName-Style.ttf
        stem = ttf.stem
        if "-" in stem:
            parts = stem.rsplit("-", 1)
            family = re.sub(r"([A-Z])", r" \1", parts[0]).strip()
            style  = parts[1]
        else:
            family = re.sub(r"([A-Z])", r" \1", stem).strip()
            style  = "Regular"
        name = f"{family} {style}".strip()
        results.append((name, str(ttf)))
    return results


def _render_text(text: str, font: ImageFont.FreeTypeFont, target_h: int) -> np.ndarray | None:
    """Render text as black-on-white, sized to target_h pixels tall."""
    # Measure
    dummy_img  = Image.new("RGB", (4000, 400), "white")
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((10, 10), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if tw <= 0 or th <= 0:
        return None

    scale  = target_h / th
    render_h = target_h + 20
    render_w = int(tw * scale) + 20

    img  = Image.new("RGB", (render_w, render_h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill="black", font=font)

    arr = np.array(img.convert("L"))
    return arr


def _normalize_for_compare(gray: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize to common dimensions and binarize."""
    resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.float32) / 255.0


def match_fonts(
    text: str,
    crop_mask: np.ndarray,       # already-extracted B&W text mask (H x W, uint8)
    font_size_px: int,
    top_n: int = 5,
) -> list[dict]:
    """
    Compare crop_mask against every available TTF render of `text`.
    Returns top_n matches sorted by similarity score.
    """
    if not text.strip() or crop_mask is None:
        return []

    fonts = _available_fonts()
    if not fonts:
        return []

    COMPARE_W, COMPARE_H = 400, 64
    crop_norm = _normalize_for_compare(crop_mask, COMPARE_W, COMPARE_H)

    results = []
    for font_name, ttf_path in fonts:
        font = _load_font(ttf_path, font_size_px * 2)
        if font is None:
            continue
        rendered = _render_text(text, font, font_size_px)
        if rendered is None:
            continue
        ref_norm = _normalize_for_compare(rendered, COMPARE_W, COMPARE_H)

        # Pixel-level similarity (inverted MSE) — fast and works well for B&W
        diff  = (crop_norm - ref_norm) ** 2
        score = float(1.0 - diff.mean())

        # Boost heavier styles since creator text is almost always bold/extra-bold
        style_boost = 1.0
        for kw, factor in _STYLE_PRIORITY.items():
            if kw.lower() in font_name.lower():
                style_boost = factor
                break
        score *= style_boost

        results.append({"font": font_name, "score": round(score, 4)})

    results.sort(key=lambda x: -x["score"])
    return results[:top_n]
