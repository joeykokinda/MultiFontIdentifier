import numpy as np
from PIL import Image
import easyocr

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _reader


def _is_ui_chrome(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> bool:
    height = y2 - y1
    # TikTok/IG UI bars: small text in the top ~8% of the image
    if y1 < img_h * 0.08 and height < 40:
        return True
    # Bottom system chrome
    if y2 > img_h * 0.95 and height < 40:
        return True
    return False


def detect_text_regions(image: Image.Image) -> list[dict]:
    reader = _get_reader()
    img_w, img_h = image.size
    img_array = np.array(image.convert("RGB"))
    results = reader.readtext(img_array)

    regions = []
    for (points, text, confidence) in results:
        pts = np.array(points, dtype=np.float32)
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())

        if (y2 - y1) < 10:
            continue
        if _is_ui_chrome(x1, y1, x2, y2, img_w, img_h):
            continue

        regions.append({
            "box": [x1, y1, x2, y2],
            "text": text,
            "confidence": float(confidence),
            "points": pts.tolist(),
        })

    return regions
