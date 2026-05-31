import cv2
import numpy as np
from PIL import Image

TARGET = 320


def extract_text_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # White/light text fill
    is_white = (r > 180) & (g > 180) & (b > 180)
    # Any saturated colored text (green, red, yellow, etc.)
    is_colored = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 120)

    mask = ((is_white | is_colored).astype(np.uint8)) * 255
    # Close holes inside letter bodies
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

    # Remove noise blobs smaller than 0.3% of image area
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = mask.size * 0.003
    clean = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    # Render: black text on white background
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

    # Pad to square preserving aspect ratio, then add 12.5% white border
    side = max(h, w)
    t, bot = (side - h) // 2, side - h - (side - h) // 2
    l, r   = (side - w) // 2, side - w - (side - w) // 2
    padded = cv2.copyMakeBorder(gray, t, bot, l, r, cv2.BORDER_CONSTANT, value=255)
    extra  = side // 8
    padded = cv2.copyMakeBorder(padded, extra, extra, extra, extra, cv2.BORDER_CONSTANT, value=255)
    resized = cv2.resize(padded, (TARGET, TARGET), interpolation=cv2.INTER_LANCZOS4)

    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))
