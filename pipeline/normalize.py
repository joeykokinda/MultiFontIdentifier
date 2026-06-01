import cv2
import numpy as np
from PIL import Image

TARGET = 320


def background_is_light(rgb: np.ndarray) -> bool:
    """Estimate background luminance from the crop border (text rarely touches it)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    b = max(1, min(2, h // 4, w // 4))
    border = np.concatenate([
        gray[:b, :].ravel(), gray[-b:, :].ravel(),
        gray[:, :b].ravel(), gray[:, -b:].ravel(),
    ])
    return float(np.median(border)) > 127


def text_pixel_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Boolean mask (True = text ink), polarity-aware.

    Works for both light-text-on-dark (TikTok/IG) and dark-text-on-light
    (e-commerce / Canva white slides). The text class is whichever side of an
    Otsu split is opposite the background; saturated pixels count as text too.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if background_is_light(rgb):
        is_lum_text = gray < thr           # dark text on light background
    else:
        is_lum_text = gray > thr           # light text on dark background

    is_colored = (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 90)
    return is_lum_text | is_colored


def extract_text_mask(rgb: np.ndarray) -> np.ndarray:
    mask = (text_pixel_mask(rgb).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

    # Remove noise blobs smaller than 0.3% of image area
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = mask.size * 0.003
    clean = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    canvas = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    canvas[clean > 0] = 0
    return canvas


def normalize_crop(image: Image.Image, box: list[int], points=None) -> Image.Image | None:
    x1, y1, x2, y2 = box
    iw, ih = image.size
    x1, y1 = max(0, x1 - 6), max(0, y1 - 6)
    x2, y2 = min(iw, x2 + 6), min(ih, y2 + 6)

    crop_rgb = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
    h, w = crop_rgb.shape[:2]
    if h < 8 or w < 8:
        return None

    gray = extract_text_mask(crop_rgb)

    side  = max(h, w)
    t, bot = (side - h) // 2, side - h - (side - h) // 2
    l, r   = (side - w) // 2, side - w - (side - w) // 2
    padded = cv2.copyMakeBorder(gray, t, bot, l, r, cv2.BORDER_CONSTANT, value=255)
    extra  = side // 8
    padded = cv2.copyMakeBorder(padded, extra, extra, extra, extra, cv2.BORDER_CONSTANT, value=255)
    resized = cv2.resize(padded, (TARGET, TARGET), interpolation=cv2.INTER_LANCZOS4)

    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))
