"""
Render-and-compare font matching (primary matcher).

Renders the OCR'd text in every downloaded TTF, then compares the rendered
letterforms against the actual image text-mask. This is the authoritative
matcher; the ViT model is only used to cluster same-font lines together.

The comparison is a DTW (dynamic time warping) distance over column ink
profiles:
  - both the crop ink and the rendered ink are tightened to their text bbox
    and height-normalized (preserving aspect),
  - each is reduced to a sequence of L2-normalized column vectors,
  - DTW aligns the two column sequences, warping out differences in letter
    spacing / tracking so the score reflects letterform *shape*, not how a
    particular font happens to space its glyphs.

This separates near-identical geometric sans siblings (League Spartan vs
Poppins vs Jost vs Urbanist) that a raw pixel-IoU cannot tell apart, because
IoU just rewards ink overlap and is dominated by spacing alignment.

Variable fonts are rendered at several weights (the wght axis is set
explicitly) so a single variable TTF can match Regular..Black; the
best-scoring weight is reported as the style.
"""
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TTF_DIR = Path(__file__).parent.parent / "fonts" / "ttf"

# Column-profile parameters.
PROFILE_H   = 40     # height every glyph strip is normalized to
MAX_COLS    = 360    # cap on column-sequence length (keeps DTW cheap)
DTW_BAND    = 28     # Sakoe-Chiba band: max column-index drift during warp
SCORE_K     = 8.0    # distance -> similarity sharpness, score = exp(-K * dist)

# Two-stage matching: a cheap low-res DTW prefilters all fonts down to
# PREFILTER_K candidates, then full-res DTW re-ranks only those. The coarse
# pass is ~15x cheaper per font and still keeps the true font near the top.
COARSE_H     = 20
COARSE_COLS  = 110
COARSE_BAND  = 14
PREFILTER_K  = 40

# Weights tried for variable fonts (wght axis value -> style name).
_VARIABLE_WEIGHTS = [
    (400, "Regular"),
    (500, "Medium"),
    (600, "SemiBold"),
    (700, "Bold"),
    (800, "ExtraBold"),
    (900, "Black"),
]


def _camel_split(stem: str) -> str:
    return re.sub(r"([A-Z])", r" \1", stem).strip()


@lru_cache(maxsize=512)
def _load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return None


def _available_fonts() -> list[tuple[str, str, str, bool]]:
    """Return (family, style, ttf_path, is_variable) for every downloaded TTF."""
    results = []
    for ttf in sorted(TTF_DIR.glob("*.ttf")):
        stem = ttf.stem
        if "-" in stem:
            family_raw, style = stem.rsplit("-", 1)
        else:
            family_raw, style = stem, "Regular"
        family = _camel_split(family_raw)
        is_variable = style.lower() in ("variable", "vf") or "[" in stem
        results.append((family, style, str(ttf), is_variable))
    return results


def _tighten(ink: np.ndarray) -> np.ndarray | None:
    """Crop an ink mask to its bounding box."""
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return None
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _profile(ink: np.ndarray) -> np.ndarray | None:
    """Height-normalize, then return L2-normalized column vectors (PROFILE_H x W)."""
    ink = _tighten(ink)
    if ink is None:
        return None
    h, w = ink.shape
    nw = max(1, min(MAX_COLS, int(round(w * PROFILE_H / h))))
    resized = cv2.resize(ink * 255, (nw, PROFILE_H), interpolation=cv2.INTER_AREA)
    cols = (resized >= 128).astype(np.float32)
    norm = np.linalg.norm(cols, axis=0, keepdims=True)
    return cols / (norm + 1e-6)


def _render_profile(text: str, font) -> np.ndarray | None:
    dummy = Image.new("L", (8000, 600), 255)
    bbox = ImageDraw.Draw(dummy).textbbox((20, 20), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None
    img = Image.new("L", (w + 40, h + 40), 255)
    ImageDraw.Draw(img).text((20 - bbox[0], 20 - bbox[1]), text, fill=0, font=font)
    ink = (np.array(img) < 128).astype(np.uint8)
    return _profile(ink)


def _coarsen(prof: np.ndarray) -> np.ndarray:
    """Downsample a column profile to a small grid for the cheap prefilter pass."""
    nw = min(COARSE_COLS, prof.shape[1])
    small = cv2.resize(prof, (nw, COARSE_H), interpolation=cv2.INTER_AREA)
    return small / (np.linalg.norm(small, axis=0, keepdims=True) + 1e-6)


def _dtw_distance(a: np.ndarray, b: np.ndarray, band: int = DTW_BAND) -> float:
    """Banded DTW over column sequences; cost is cosine distance per column pair."""
    cost = 1.0 - (a.T @ b)            # Wa x Wb, in [0, 2]
    wa, wb = cost.shape
    INF = 1e9
    acc = np.full((wa + 1, wb + 1), INF)
    acc[0, 0] = 0.0
    for i in range(1, wa + 1):
        lo = max(1, i - band)
        hi = min(wb, i + band)
        row, prev, c = acc[i], acc[i - 1], cost[i - 1]
        for j in range(lo, hi + 1):
            row[j] = c[j - 1] + min(prev[j], row[j - 1], prev[j - 1])
    return float(acc[wa, wb] / (wa + wb))


def match_fonts(text: str, crop_mask: np.ndarray, font_size_px: int,
                top_n: int = 5) -> list[dict]:
    """
    Compare the crop's text mask against every available TTF render of `text`.
    Returns top_n candidates as {"font", "style", "score", "raw"} sorted by score.
    `crop_mask` is white-background (255) / black-text (0), as produced by
    pipeline.normalize.extract_text_mask.
    """
    if not text.strip() or crop_mask is None:
        return []
    fonts = _available_fonts()
    if not fonts:
        return []

    crop_profile = _profile((crop_mask < 128).astype(np.uint8))
    if crop_profile is None:
        return []
    crop_coarse = _coarsen(crop_profile)

    render_size = 120

    # --- Stage 1: render every font/weight, keep each font's best weight by
    #     a cheap coarse DTW. Collapse to one entry per family. ---
    by_family: dict[str, dict] = {}
    for family, style, ttf_path, is_variable in fonts:
        font = _load_font(ttf_path, render_size)
        if font is None:
            continue

        trials = _VARIABLE_WEIGHTS if is_variable else [(None, style)]

        best = None
        for wght, style_name in trials:
            if wght is not None:
                try:
                    font.set_variation_by_axes([wght])
                except Exception:
                    # Not a single-axis wght font; render its default once.
                    if best is not None:
                        break
                    style_name = style
            prof = _render_profile(text, font)
            if prof is None:
                continue
            coarse_dist = _dtw_distance(crop_coarse, _coarsen(prof), band=COARSE_BAND)
            if best is None or coarse_dist < best["coarse"]:
                best = {"coarse": coarse_dist, "profile": prof, "style": style_name,
                        "family": family, "raw": Path(ttf_path).stem}

        if best is None:
            continue
        cur = by_family.get(family)
        if cur is None or best["coarse"] < cur["coarse"]:
            by_family[family] = best

    # --- Stage 2: full-res DTW re-rank of the top prefiltered candidates. ---
    prefiltered = sorted(by_family.values(), key=lambda x: x["coarse"])[:PREFILTER_K]
    for m in prefiltered:
        m["dist"] = _dtw_distance(crop_profile, m["profile"])

    ranked = sorted(prefiltered, key=lambda x: x["dist"])
    out = []
    for m in ranked[:top_n]:
        out.append({
            "font":  m["family"],
            "style": m["style"],
            "score": round(float(np.exp(-SCORE_K * m["dist"])), 4),
            "raw":   m["raw"],
        })
    return out
