"""
Typographic layout extraction.

Recovers the properties you'd need to recreate a caption in an editor:
  - position of each line / block in the image,
  - text alignment (left / center / right) within a multi-line block,
  - line spacing (leading) in px and as a ratio of the font size,
  - the true font size in px (fitted by rendering the matched font), and
  - letter spacing / tracking (em), measured against the matched font's
    natural advance widths.

Spacing and size measurements need the matched font; alignment, position and
line spacing are purely geometric and work even when the font is unknown.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline.match import _load_font, TTF_DIR
from pipeline.normalize import text_pixel_mask

_STYLE_WGHT = {
    "Thin": 100, "ExtraLight": 200, "Light": 300, "Regular": 400,
    "Medium": 500, "SemiBold": 600, "Bold": 700, "ExtraBold": 800, "Black": 900,
}


def _font(raw: str, style: str, size: int):
    """Load the matched TTF at `size`, setting the wght axis for variable fonts."""
    if not raw:
        return None
    path = str(TTF_DIR / f"{raw}.ttf")
    font = _load_font(path, size)
    if font is None:
        return None
    if raw.endswith("-Variable"):
        try:
            font.set_variation_by_axes([_STYLE_WGHT.get(style, 700)])
        except Exception:
            pass
    return font


def _ink_bbox(image: Image.Image, box: list[int]) -> tuple[int, int] | None:
    """Tight (width, height) of the actual text ink inside an OCR box."""
    x1, y1, x2, y2 = box
    crop = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
    if crop.size == 0:
        return None
    ys, xs = np.where(text_pixel_mask(crop))
    if len(xs) == 0:
        return None
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _rendered_ink(text: str, font) -> tuple[int, int]:
    """(width, height) of text ink rendered with the font's natural spacing."""
    dummy = Image.new("L", (8000, 600), 255)
    bb = ImageDraw.Draw(dummy).textbbox((20, 20), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def fit_font_size(text: str, raw: str, style: str, target_h: int) -> int:
    """Binary-search the font px size whose rendered ink height matches the crop."""
    lo, hi = 6, 400
    best = target_h
    for _ in range(12):
        mid = (lo + hi) // 2
        font = _font(raw, style, mid)
        if font is None:
            return target_h
        _, h = _rendered_ink(text, font)
        if h <= 0:
            break
        best = mid
        if h < target_h:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _ink_mask(image: Image.Image, box: list[int]) -> np.ndarray | None:
    crop = np.array(image.crop(tuple(box)).convert("RGB"))
    if crop.size == 0:
        return None
    return text_pixel_mask(crop).astype(np.uint8)


def _column_gaps(ink: np.ndarray) -> list[int]:
    """Widths of empty-column runs between ink (inter-glyph and inter-word gaps)."""
    col = ink.any(axis=0)
    xs = np.where(col)[0]
    if len(xs) == 0:
        return []
    col = col[xs.min():xs.max() + 1]       # trim leading/trailing blank
    gaps, run = [], 0
    for c in col:
        if c:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    return gaps


def word_spacing(image: Image.Image, box: list[int], size: int) -> dict:
    """
    Median inter-word gap in px (and em). Word gaps are the wide cluster of
    empty-column runs; narrow runs are inter-letter spacing.
    """
    ink = _ink_mask(image, box)
    if ink is None:
        return {}
    gaps = _column_gaps(ink)
    if len(gaps) < 2:
        return {}
    gaps_sorted = sorted(gaps)
    med = gaps_sorted[len(gaps_sorted) // 2]
    word_gaps = [g for g in gaps if g > max(2.0 * med, med + 3)]
    if not word_gaps:
        return {"word_spacing_px": None}
    wg = float(np.median(word_gaps))
    out = {"word_spacing_px": round(wg, 1)}
    if size:
        out["word_spacing_em"] = round(wg / size, 3)
    return out


def rotation_deg(points) -> float | None:
    """Baseline rotation from the OCR quad's top edge (deg, + = clockwise down)."""
    if not points or len(points) < 2:
        return None
    (x0, y0), (x1, y1) = points[0], points[1]
    return round(float(np.degrees(np.arctan2(y1 - y0, x1 - x0))), 1)


def slant_deg(image: Image.Image, box: list[int]) -> float | None:
    """
    Faux-italic slant via shear search: shear the ink to maximize vertical
    stroke alignment; the shear that uprights it is the negative of the slant.
    """
    ink = _ink_mask(image, box)
    if ink is None or ink.sum() == 0:
        return None
    h, w = ink.shape
    ys = np.where(ink.any(axis=1))[0]
    cap_h = ys.max() - ys.min() + 1
    best_angle, best_score = 0.0, -1.0
    for angle in range(-24, 25, 2):
        shift = np.tan(np.radians(angle))
        M = np.array([[1, shift, -shift * h / 2], [0, 1, 0]], dtype=np.float32)
        sheared = cv2.warpAffine(ink, M, (w, h))
        colsum = sheared.sum(axis=0).astype(np.float64)
        score = float((colsum ** 2).sum())   # sharper columns => taller vertical strokes
        if score > best_score:
            best_score, best_angle = score, angle
    if cap_h < 12 or abs(best_angle) <= 2:   # deadband: ignore sub-2° noise
        return 0.0
    return float(-best_angle)


def analyze_line(image: Image.Image, box: list[int], text: str,
                 raw: str | None, style: str, points=None) -> dict:
    """Per-line metrics: fitted px size, ink height, letter/word spacing, slant."""
    out: dict = {}
    ink = _ink_bbox(image, box)
    if ink is None:
        return out
    ink_w, ink_h = ink
    out["text_ink_height_px"] = ink_h
    out["rotation_deg"] = rotation_deg(points)
    out["slant_deg"] = slant_deg(image, box)

    if not raw or not text.strip():
        return out

    size = fit_font_size(text, raw, style, ink_h)
    out["font_size_px"] = size

    font = _font(raw, style, size)
    if font is None:
        return out
    natural_w, _ = _rendered_ink(text, font)
    n_gaps = max(1, len(text) - 1)
    extra_per_gap = (ink_w - natural_w) / n_gaps
    out["letter_spacing_px"] = round(extra_per_gap, 1)
    out["letter_spacing_em"] = round(extra_per_gap / size, 3) if size else 0.0
    out.update(word_spacing(image, box, size))
    return out


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _x_overlap(a: list[int], b: list[int]) -> float:
    """Fraction of the narrower box's width that overlaps horizontally with the other."""
    lo, hi = max(a[0], b[0]), min(a[2], b[2])
    inter = max(0, hi - lo)
    narrow = max(1, min(a[2] - a[0], b[2] - b[0]))
    return inter / narrow


def group_blocks(regions: list[dict], image_w: int, image_h: int) -> list[dict]:
    """
    Group OCR lines into visual paragraphs by vertical adjacency + horizontal
    overlap, independent of font. Returns a block per paragraph with alignment,
    line spacing and position — the geometry needed to recreate the layout.
    """
    items = sorted(regions, key=lambda r: r["box"][1])
    blocks: list[list[dict]] = []
    for r in items:
        placed = False
        for blk in blocks:
            prev = blk[-1]["box"]
            line_h = prev[3] - prev[1]
            gap = r["box"][1] - prev[3]
            # next line sits just below and shares horizontal extent
            if -0.4 * line_h <= gap <= 1.2 * line_h and _x_overlap(prev, r["box"]) > 0.25:
                blk.append(r)
                placed = True
                break
        if not placed:
            blocks.append([r])

    out = []
    for blk in blocks:
        instances = [{"region": r["box"]} for r in blk]
        info = analyze_block(instances, image_w, image_h)
        info["line_count"] = len(blk)
        align = info.get("text_align", "left")
        bx1, _, bx2, _ = info["block_region"]
        bcx = (bx1 + bx2) / 2
        lines = []
        for r in blk:
            x1, _, x2, _ = r["box"]
            if align == "right":
                offset = bx2 - x2
            elif align == "center":
                offset = round((x1 + x2) / 2 - bcx, 1)
            else:
                offset = x1 - bx1
            lines.append({"region": r["box"], "text": r.get("text", ""),
                          "offset_from_block_px": round(float(offset), 1)})
        info["lines"] = lines
        out.append(info)
    return out


def analyze_block(instances: list[dict], image_w: int, image_h: int) -> dict:
    """Block-level geometry: alignment, line spacing, and position in the image."""
    boxes = [i["region"] for i in instances]
    if not boxes:
        return {}

    lefts   = [b[0] for b in boxes]
    rights  = [b[2] for b in boxes]
    centers = [(b[0] + b[2]) / 2 for b in boxes]

    block = [min(lefts), min(b[1] for b in boxes),
             max(rights), max(b[3] for b in boxes)]

    out: dict = {"block_region": block}

    if len(boxes) >= 2:
        # Alignment = the edge whose positions vary least across lines.
        spreads = {"left": _spread(lefts), "center": _spread(centers), "right": _spread(rights)}
        out["text_align"] = min(spreads, key=spreads.get)

        tops = sorted(b[1] for b in boxes)
        gaps = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
        line_spacing = float(np.median(gaps))
        out["line_spacing_px"] = round(line_spacing, 1)
        heights = [b[3] - b[1] for b in boxes]
        med_h = float(np.median(heights))
        if med_h > 0:
            out["line_height_ratio"] = round(line_spacing / med_h, 2)
    else:
        # Single line: infer alignment from the image margins.
        b = boxes[0]
        left_margin, right_margin = b[0], image_w - b[2]
        if abs(left_margin - right_margin) <= 0.06 * image_w:
            out["text_align"] = "center"
        elif left_margin < right_margin:
            out["text_align"] = "left"
        else:
            out["text_align"] = "right"

    # Coarse position of the block within the image.
    cx = (block[0] + block[2]) / 2 / image_w
    cy = (block[1] + block[3]) / 2 / image_h
    out["horizontal_position"] = ("left" if cx < 0.4 else "right" if cx > 0.6 else "center")
    out["vertical_position"]   = ("top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle")
    return out
