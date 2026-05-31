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

COPY . .

# Pre-download models at build time so startup is instant
RUN python -c "\
import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False); \
from huggingface_hub import hf_hub_download; \
import yaml; \
hf_hub_download('storia/font-classify-onnx', 'model.onnx',         cache_dir='models'); \
hf_hub_download('storia/font-classify-onnx', 'model_config.yaml',  cache_dir='models'); \
"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
