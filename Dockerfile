FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time so startup is fast
COPY . .
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=False, lang='en', show_log=False)"
RUN python -c "from transformers import ViTForImageClassification, ViTImageProcessor; \
    ViTImageProcessor.from_pretrained('Storia-AI/font-classify', cache_dir='models'); \
    ViTForImageClassification.from_pretrained('Storia-AI/font-classify', cache_dir='models')"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
