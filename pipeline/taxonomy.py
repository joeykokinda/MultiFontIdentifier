"""
Typographic classification.

For ambiguous matches the family is far more certain than the exact font: a
Garamond-style serif is an easy call even when Cormorant vs Cardo vs EB
Garamond is a coin-flip. This module names the category/style of a detected
font and reports two separate confidences:
  - family_confidence: how consistent the top candidates are in category, and
  - match_confidence:  how clearly the single best font wins (the verdict).

Serif vs sans vs slab and stroke contrast are measured from the rendered
glyphs (no font database needed); curated keywords add the finer sub-label
(Garamond-style, Didone, Geometric, etc.).
"""
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline.layout import _font

# Curated sub-labels keyed by family name. First match wins.
_KEYWORD_STYLES = [
    (r"cormorant|garamond|cardo|lusitana|sorts|gentium|vollkorn|spectral", "Old-style serif (Garamond-style)"),
    (r"playfair|bodoni|didot|abril|ultra", "Didone / Modern serif"),
    (r"slab|rokkitt|arvo|zilla|josefin\s*slab|alfa", "Slab serif"),
    (r"merriweather|pt\s*serif|lora|noto\s*serif|source\s*serif|crimson|bitter", "Transitional serif"),
    (r"poppins|montserrat|league\s*spartan|jost|futura|century\s*gothic|questrial|outfit|urbanist", "Geometric sans-serif"),
    (r"baloo|fredoka|varela\s*round|quicksand|comfortaa|nunito\b|rowdies", "Rounded sans-serif"),
    (r"open\s*sans|lato|source\s*sans|pt\s*sans|nunito\s*sans|hind|mukta|assistant", "Humanist sans-serif"),
    (r"roboto|inter|helvet|arial|work\s*sans|archivo|figtree|manrope|sora", "Grotesque sans-serif"),
    (r"anton|bebas|oswald|staatliches|teko|saira\s*condensed|barlow\s*condensed", "Condensed display"),
    (r"pacifico|lobster|dancing|kaushan|caveat|satisfy|sacramento|great\s*vibes|allura", "Script"),
    (r"permanent\s*marker|indie\s*flower|shadows\s*into|patrick\s*hand|architects|reenie|handlee|rock\s*salt|covered\s*by", "Handwriting"),
    (r"mono|share\s*tech|courier|space\s*mono|jetbrains|inconsolata", "Monospace"),
]

_FAMILY = {
    "Old-style serif (Garamond-style)": "serif", "Didone / Modern serif": "serif",
    "Transitional serif": "serif", "Slab serif": "slab-serif",
    "Geometric sans-serif": "sans-serif", "Rounded sans-serif": "sans-serif",
    "Humanist sans-serif": "sans-serif", "Grotesque sans-serif": "sans-serif",
    "Condensed display": "display", "Script": "script",
    "Handwriting": "handwriting", "Monospace": "monospace",
}


def _glyph_ink(raw: str, style: str, ch: str, size: int = 160):
    font = _font(raw, style, size)
    if font is None:
        return None
    bb = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), ch, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    if w <= 0 or h <= 0:
        return None
    img = Image.new("L", (w + 20, h + 20), 255)
    ImageDraw.Draw(img).text((10 - bb[0], 10 - bb[1]), ch, fill=0, font=font)
    ink = (np.array(img) < 128).astype(np.uint8)
    ys, xs = np.where(ink)
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _measure(raw: str, style: str) -> dict:
    """Render glyphs to gauge serif presence and stroke contrast."""
    out = {"serif": None, "contrast": None}
    cap = _glyph_ink(raw, style, "I")
    if cap is not None and cap.shape[0] > 8:
        h = cap.shape[0]
        band = max(1, h // 8)
        widths = cap.sum(axis=1)
        foot = float(max(widths[:band].max(), widths[-band:].max()))
        stem = float(np.median(widths[h // 3: 2 * h // 3]))
        out["serif"] = foot / stem if stem > 0 else None
    bowl = _glyph_ink(raw, style, "o")
    if bowl is not None and bowl.sum() > 0:
        dt = cv2.distanceTransform((bowl * 255).astype(np.uint8), cv2.DIST_L2, 5)
        vals = 2.0 * dt[bowl > 0]
        if len(vals) > 0:
            thin = np.percentile(vals, 35)
            thick = np.percentile(vals, 95)
            out["contrast"] = float(thick / thin) if thin > 0 else None
    return out


def classify_font(raw: str | None, style: str = "Regular") -> dict:
    """Return {category, style_class} for one font."""
    if not raw:
        return {"category": "unknown", "style_class": "unknown"}
    for pattern, label in _KEYWORD_STYLES:
        if re.search(pattern, raw, re.I):
            return {"category": _FAMILY[label], "style_class": label}

    # No keyword: fall back to rendered-glyph heuristics.
    m = _measure(raw, style)
    serif, contrast = m["serif"], m["contrast"]
    if serif is None:
        return {"category": "unknown", "style_class": "unknown"}
    if serif < 1.5:
        return {"category": "sans-serif", "style_class": "sans-serif"}
    if contrast is not None and contrast >= 2.8:
        return {"category": "serif", "style_class": "Didone / Modern serif"}
    if contrast is not None and contrast < 1.4:
        return {"category": "slab-serif", "style_class": "Slab serif"}
    return {"category": "serif", "style_class": "serif"}


_LABEL = {"verified": "high", "likely": "medium", "uncertain": "low"}


def classify_detection(top5: list[dict], verdict: dict) -> dict:
    """
    Family classification for a detected font: the style class of the best
    match, how confident the family is (agreement across candidates), and how
    confident the exact font is (the match verdict).
    """
    if not top5:
        return {}
    best = classify_font(top5[0].get("raw"), top5[0].get("style", "Regular"))
    cats = [classify_font(c.get("raw"), c.get("style", "Regular"))["category"]
            for c in top5]
    known = [c for c in cats if c != "unknown"]
    if known:
        agree = known.count(best["category"]) / len(known)
        fam_conf = "high" if agree >= 0.8 else "medium" if agree >= 0.5 else "low"
    else:
        fam_conf = "low"

    return {
        "category": best["category"],
        "style_class": best["style_class"],
        "family_confidence": fam_conf,
        "match_confidence": _LABEL.get(verdict.get("verdict"), "low"),
        "best_match": top5[0]["font"],
        "alternatives": verdict.get("alternatives", []),
    }
