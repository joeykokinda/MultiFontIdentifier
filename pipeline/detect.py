import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _ocr


def detect_text_regions(image: Image.Image) -> list[dict]:
    ocr = _get_ocr()
    img_array = np.array(image.convert("RGB"))
    result = ocr.ocr(img_array, cls=False)

    regions = []
    if not result or not result[0]:
        return regions

    for line in result[0]:
        points, (text, confidence) = line
        pts = np.array(points, dtype=np.float32)
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())
        if (y2 - y1) < 10:
            continue
        regions.append({
            "box": [x1, y1, x2, y2],
            "text": text,
            "confidence": float(confidence),
            "points": pts.tolist(),
        })

    return regions
