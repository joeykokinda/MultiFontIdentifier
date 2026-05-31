import cv2
import numpy as np
from PIL import Image


def detect_background(image: Image.Image, box: list[int]) -> dict:
    x1, y1, x2, y2 = box
    img_w, img_h = image.size

    inner = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
    if inner.size == 0:
        return {"background": "none", "background_color": None}

    gray = cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY)
    flat = np.sort(gray.flatten())

    # Bottom 20% = background pixels, top 20% = text/foreground pixels
    split = max(1, len(flat) // 5)
    bg_brightness = float(np.median(flat[:split]))
    fg_brightness = float(np.median(flat[-split:]))
    contrast = fg_brightness - bg_brightness

    if contrast < 50:
        # No clear box behind the text
        return {"background": "none", "background_color": None}

    if bg_brightness < 60:
        return {"background": "black_box", "background_color": "#000000"}
    if bg_brightness > 200:
        return {"background": "white_box", "background_color": "#FFFFFF"}

    # Colored box: sample the background pixels
    mask = gray < (bg_brightness + 30)
    bg_pixels = inner[mask]
    if len(bg_pixels) == 0:
        return {"background": "colored_box", "background_color": None}

    r, g, b = bg_pixels.mean(axis=0).astype(int)
    return {"background": "colored_box", "background_color": f"#{r:02X}{g:02X}{b:02X}"}


def detect_outline(image: Image.Image, box: list[int]) -> dict:
    x1, y1, x2, y2 = box
    crop = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))

    if crop.size == 0:
        return {"has_outline": False, "outline_color": None}

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    outline_mask = dilated & ~binary

    outline_pixels = crop[outline_mask > 0]
    text_pixels = crop[binary > 0]

    if len(outline_pixels) == 0 or len(text_pixels) == 0:
        return {"has_outline": False, "outline_color": None}

    mean_outline = outline_pixels.mean(axis=0)
    mean_text = text_pixels.mean(axis=0)
    color_diff = float(np.abs(mean_outline.astype(float) - mean_text.astype(float)).mean())

    if color_diff < 40:
        return {"has_outline": False, "outline_color": None}

    r, g, b = mean_outline.astype(int)
    return {"has_outline": True, "outline_color": f"#{r:02X}{g:02X}{b:02X}"}


def detect_text_color(image: Image.Image, box: list[int]) -> str:
    x1, y1, x2, y2 = box
    crop = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))

    if crop.size == 0:
        return "#000000"

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Text pixels are either the light or dark ones depending on background
    bg_mean = float(gray[binary == 0].mean()) if (binary == 0).any() else 128
    text_mask = binary > 0 if bg_mean < 128 else binary == 0
    text_pixels = crop[text_mask]

    if len(text_pixels) == 0:
        return "#000000"

    r, g, b = text_pixels.mean(axis=0).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"
