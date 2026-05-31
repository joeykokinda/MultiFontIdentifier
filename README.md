# MultiFontIdentifier

Open-source font detection for TikTok, Instagram, and Canva-style slideshows. Send any image, get back every font on it: name, style, size, location, text color, background boxes, bold/italic/outline effects.

## How it works

```
image → text detection (PaddleOCR) → crop normalization → font classification (ViT)
     → embedding clustering → effect analysis → JSON result
```

For each detected text region the tool returns:
- Font family and style (Bold, Italic, ExtraBold, etc.)
- Estimated font size in pixels
- Bounding box location `[x1, y1, x2, y2]`
- Text color as hex
- Background type: `black_box`, `white_box`, `colored_box`, or `none`
- Whether the text has an outline and its color
- The raw detected text

## Quickstart (CLI)

```bash
pip install -r requirements.txt

# Drop any image into images/ and run
python detect.py images/slide.png
```

## Quickstart (API server)

```bash
uvicorn api.main:app --reload
```

```bash
curl -F "file=@images/slide.png" http://localhost:8000/detect
```

## Docker

```bash
docker build -t multifontidentifier .
docker run -p 8000:8000 multifontidentifier
```

## Example response

```json
{
  "fonts_detected": [
    {
      "font": "Montserrat",
      "style": "ExtraBold",
      "is_italic": false,
      "confidence": 0.91,
      "source": "canva_popular",
      "instances": [
        {
          "region": [120, 80, 640, 150],
          "font_size_px": 70,
          "text": "THIS IS THE HEADLINE",
          "background": "black_box",
          "background_color": "#000000",
          "text_color": "#FFFFFF",
          "has_outline": false,
          "outline_color": null
        }
      ]
    },
    {
      "font": "Playfair Display",
      "style": "Italic",
      "is_italic": true,
      "confidence": 0.84,
      "source": "canva_popular",
      "instances": [
        {
          "region": [180, 230, 710, 290],
          "font_size_px": 32,
          "text": "and this is the subtext",
          "background": "none",
          "background_color": null,
          "text_color": "#F5F5F5",
          "has_outline": true,
          "outline_color": "#000000"
        }
      ]
    }
  ],
  "unknown_regions": []
}
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/detect` | Multipart image upload (`file` field) |
| `POST` | `/detect/base64` | JSON body `{"image": "<base64>"}` |
| `GET` | `/fonts` | List all fonts in the database |
| `GET` | `/health` | Liveness check |

## Font databases

| File | Source | Count |
|------|--------|-------|
| `fonts/tiktok.json` | TikTok native in-app fonts | ~12 |
| `fonts/capcut.json` | CapCut font library | ~65 |
| `fonts/canva_popular.json` | Most-used Canva fonts | ~110 |
| `fonts/alias.json` | Platform name to Google Font mapping | ~30 |

## Model

Font classification uses `Storia-AI/font-classify`, a ViT model trained on ~3,000 Google Fonts. Set `FONT_MODEL` env var to swap in a different HuggingFace model.

## Known limitations

- Heavily stylized or glow-effect text reduces classification confidence
- Private/custom fonts not in Google Fonts return the nearest visual match
- Sub-10px text is filtered out (too small to classify reliably)
- Exact font variant (e.g., SemiBold vs Bold) may differ; family name is reliable
