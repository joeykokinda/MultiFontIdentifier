import cv2
import numpy as np
from PIL import Image


def _text_fill_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    is_white   = (r > 150) & (g > 150) & (b > 150)
    is_colored = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 100)
    mask = ((is_white | is_colored).astype(np.uint8)) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def detect_outline(image: Image.Image, box: list[int]) -> dict:
    x1, y1, x2, y2 = box
    iw, ih = image.size
    x1, y1 = max(0, x1 - 4), max(0, y1 - 4)
    x2, y2 = min(iw, x2 + 4), min(ih, y2 + 4)
    crop = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))

    if crop.size == 0:
        return {"has_outline": False, "outline_color": None}

    fill_mask = _text_fill_mask(crop)

    if fill_mask.sum() == 0:
        return {"has_outline": False, "outline_color": None}

    # Dilate the fill region to get the "halo" zone just outside the letters
    kernel = np.ones((4, 4), np.uint8)
    dilated = cv2.dilate(fill_mask, kernel, iterations=2)
    outline_zone = (dilated > 0) & (fill_mask == 0)

    outline_pixels = crop[outline_zone]
    fill_pixels    = crop[fill_mask > 0]

    if len(outline_pixels) < 10 or len(fill_pixels) < 10:
        return {"has_outline": False, "outline_color": None}

    mean_outline = outline_pixels.mean(axis=0)
    mean_fill    = fill_pixels.mean(axis=0)

    # Outline exists when the halo zone is significantly darker than the text fill
    brightness_diff = float(mean_fill.mean()) - float(mean_outline.mean())
    has_outline = brightness_diff > 50

    if not has_outline:
        return {"has_outline": False, "outline_color": None}

    r, g, b = mean_outline.astype(int)
    return {"has_outline": True, "outline_color": f"#{r:02X}{g:02X}{b:02X}"}


def detect_background(image: Image.Image, box: list[int]) -> dict:
    x1, y1, x2, y2 = box
    iw, ih = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)

    inner = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
    if inner.size == 0:
        return {"background": "none", "background_color": None}

    gray  = cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY)
    flat  = np.sort(gray.flatten())
    split = max(1, len(flat) // 5)
    bg_brightness = float(np.median(flat[:split]))
    fg_brightness = float(np.median(flat[-split:]))
    contrast      = fg_brightness - bg_brightness

    if contrast < 50:
        return {"background": "none", "background_color": None}

    if bg_brightness < 60:
        return {"background": "black_box", "background_color": "#000000"}
    if bg_brightness > 200:
        return {"background": "white_box", "background_color": "#FFFFFF"}

    mask      = gray < (bg_brightness + 30)
    bg_pixels = inner[mask]
    if len(bg_pixels) == 0:
        return {"background": "colored_box", "background_color": None}

    r, g, b = bg_pixels.mean(axis=0).astype(int)
    return {"background": "colored_box", "background_color": f"#{r:02X}{g:02X}{b:02X}"}


def detect_text_color(image: Image.Image, box: list[int]) -> str:
    x1, y1, x2, y2 = box
    iw, ih = image.size
    crop = np.array(image.crop((max(0, x1), max(0, y1), min(iw, x2), min(ih, y2))).convert("RGB"))

    if crop.size == 0:
        return "#FFFFFF"

    fill_mask = _text_fill_mask(crop)
    text_px   = crop[fill_mask > 0]

    if len(text_px) == 0:
        return "#FFFFFF"

    r, g, b = text_px.mean(axis=0).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"
