import cv2
import numpy as np
from PIL import Image

TARGET_HEIGHT = 64


def normalize_crop(image: Image.Image, box: list[int], points: list | None = None) -> Image.Image | None:
    x1, y1, x2, y2 = box
    img_w, img_h = image.size

    pad = 4
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_w, x2 + pad)
    y2 = min(img_h, y2 + pad)

    crop = image.crop((x1, y1, x2, y2))
    w, h = crop.size
    if h == 0 or w == 0:
        return None

    gray = np.array(crop.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    scale = TARGET_HEIGHT / h
    new_w = max(1, int(w * scale))
    result = Image.fromarray(enhanced).resize((new_w, TARGET_HEIGHT), Image.LANCZOS)

    return result.convert("RGB")
