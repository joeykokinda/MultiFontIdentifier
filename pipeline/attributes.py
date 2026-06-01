"""
Per-line type attributes: casing, decoration (underline / strikethrough),
numeric weight, and width class.

Casing comes from the OCR string. Decoration, weight and width are measured
from the text ink (and, for weight/width, calibrated against the matched
font so the numbers mean the same thing across families).
"""
import re

import numpy as np
from PIL import Image

from pipeline.normalize import text_pixel_mask

_STYLE_WGHT = {
    "Thin": 100, "ExtraLight": 200, "Light": 300, "Regular": 400,
    "Medium": 500, "SemiBold": 600, "Bold": 700, "ExtraBold": 800, "Black": 900,
}


def text_case(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "none"
    if all(c.isupper() for c in letters):
        return "upper"
    if all(c.islower() for c in letters):
        return "lower"
    words = [w for w in re.split(r"\s+", text) if any(c.isalpha() for c in w)]
    if words and all(w[0].isupper() and w[1:].islower() for w in words if len(w) > 1):
        return "title"
    # First alpha upper, rest lower -> sentence case.
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return "sentence"
    return "mixed"


def _ink(image: Image.Image, box: list[int]) -> np.ndarray | None:
    crop = np.array(image.crop(tuple(box)).convert("RGB"))
    if crop.size == 0:
        return None
    return text_pixel_mask(crop).astype(np.uint8)


def _longest_run(row: np.ndarray) -> int:
    """Longest run of consecutive ink in a single row."""
    best = run = 0
    for v in row:
        run = run + 1 if v else 0
        if run > best:
            best = run
    return best


def detect_decoration(image: Image.Image, box: list[int]) -> dict:
    """
    A rule is a THIN band of rows that each span the text width as one
    continuous run (glyphs always leave inter-letter gaps, so dense bold text
    never does). Underline vs strikethrough is decided by ink mass: an
    underline has almost all letter ink above it (only descenders below); a
    strikethrough cuts through, with ink on both sides.
    """
    ink = _ink(image, box)
    out = {"underline": False, "strikethrough": False}
    if ink is None or ink.sum() == 0:
        return out
    h = ink.shape[0]
    xs = np.where(ink.any(axis=0))[0]
    if len(xs) == 0:
        return out
    span = xs.max() - xs.min() + 1
    lo, hi = xs.min(), xs.max() + 1

    bar_rows = [y for y in range(h) if _longest_run(ink[y, lo:hi]) / span >= 0.72]
    if not bar_rows:
        return out

    # Group consecutive bar rows into bands; a real rule is thin.
    bands, cur = [], [bar_rows[0]]
    for y in bar_rows[1:]:
        if y == cur[-1] + 1:
            cur.append(y)
        else:
            bands.append(cur); cur = [y]
    bands.append(cur)

    total = float(ink.sum())
    for band in bands:
        if len(band) > max(6, 0.18 * h):     # too thick to be a rule (it's text)
            continue
        top, bot = band[0], band[-1]
        above = float(ink[:top].sum())
        below = float(ink[bot + 1:].sum())
        if above + below < 0.15 * total:
            continue
        if below < 0.15 * above:
            out["underline"] = True
        elif below > 0.30 * above:
            out["strikethrough"] = True
    return out


def fit_weight(style: str, raw: str | None) -> dict:
    """
    Numeric weight 100-900 from the style the matcher selected. The render-
    and-compare step already chose this weight by comparing letterform shape
    against the image, which is more reliable than re-measuring stroke width
    on a noisy text mask.
    """
    if not raw:
        return {}
    return {"weight": _STYLE_WGHT.get(style, 400)}


def width_class(raw: str | None) -> dict:
    """
    Width class from the matched font's own category. Render-and-compare picks
    the font whose glyph width fits the image, so the matched family's inherent
    width is the answer (measuring width from ink is confounded by tracking).
    """
    if not raw:
        return {}
    if re.search(r"condensed|narrow", raw, re.I):
        return {"width_class": "condensed"}
    if re.search(r"expanded|extended|wide", raw, re.I):
        return {"width_class": "expanded"}
    return {"width_class": "normal"}
