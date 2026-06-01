# MultiFontIdentifier

Open-source font detection for TikTok, Instagram, and Canva-style slideshows. Send any image, get back every font on it: family name, style, size, location, text color, outline color, background box type.

Built for agents that need to programmatically identify fonts in social media content.

---

## Setup (run once)

```bash
# 1. Clone
git clone https://github.com/joeykokinda/MultiFontIdentifier.git
cd MultiFontIdentifier

# 2. Create venv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Download all font TTFs (pulls from Google Fonts via git, no API key needed)
python scripts/download_fonts.py

# 4. Models download automatically on first run (EasyOCR + storia/font-classify-onnx)
```

**Optional — add TikTok Sans** (their proprietary caption font, not on Google Fonts):
Download from TikTok's brand site and place the file at `fonts/ttf/TikTokSans-Bold.ttf`. The pipeline picks it up automatically.

---

## Usage

### CLI

Drop an image in `images/` and run:

```bash
python detect.py images/slide.png
```

Output is JSON to stdout.

### API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t multifontidentifier .
docker run -p 8000:8000 multifontidentifier
```

---

## API reference

### `POST /detect` — detect all fonts in an image

**Request (multipart):**
```bash
curl -F "file=@slide.png" http://localhost:8000/detect
```

**Request (base64):**
```bash
curl -X POST http://localhost:8000/detect/base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>"}'
```

**Response:**
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

### Response field reference

| Field | Type | Description |
|-------|------|-------------|
| `font` | string | Font family name (e.g. `"Montserrat"`, `"TikTok Typewriter"`) |
| `style` | string | Weight/style variant: `Bold`, `ExtraBold`, `Black`, `SemiBold`, `Regular` |
| `is_italic` | bool | Whether the text is italicized |
| `confidence` | float | 0.0–1.0. Above 0.7 is high confidence. 0.3–0.7 is a best estimate |
| `source` | string | Which database matched: `tiktok`, `capcut`, `canva_popular`, `google_fonts` |
| `nearest_google_font` | string | Only present for TikTok native fonts. The closest Google Font equivalent |
| `instances[].region` | int[4] | Bounding box `[x1, y1, x2, y2]` in pixels |
| `instances[].font_size_px` | int | Approximate cap height in pixels (bounding box height) |
| `instances[].text` | string | The detected text content in this region |
| `instances[].background` | string | `"black_box"`, `"white_box"`, `"colored_box"`, or `"none"` |
| `instances[].background_color` | string or null | Hex color of the background box, e.g. `"#000000"` |
| `instances[].text_color` | string | Hex color of the text fill, e.g. `"#FFFFFF"` |
| `instances[].has_outline` | bool | Whether the text has a stroke/outline effect |
| `instances[].outline_color` | string or null | Hex color of the outline, e.g. `"#1A1A1A"` |

### `GET /fonts` — list all known fonts

Returns every font in the database with its source tag.

```bash
curl http://localhost:8000/fonts
```

### `GET /health` — liveness check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## How classification works

Two-pass pipeline:

1. **EasyOCR** detects all text bounding boxes and reads the text content.
2. **storia/font-classify-onnx** (EfficientNet-B3, 3,473 Google Fonts) classifies each cropped region.
3. **Render-and-compare**: the detected text is rendered in every font TTF in `fonts/ttf/` and compared pixel-by-pixel against the real crop. This gives exact matches for any font we have a TTF for.
4. **Effect analysis** runs per-region: background box color, text fill color, outline detection.

Pass 3 (render-compare) takes priority over Pass 2 (ViT model) when TTF files are present. The more fonts in `fonts/ttf/`, the more accurate the results.

---

## Font coverage

| Platform | Coverage | Notes |
|----------|----------|-------|
| CapCut | ~95% | Nearly all CapCut fonts are Google Fonts |
| Canva templates | ~90% | Top 110 Canva fonts covered |
| TikTok in-app text | ~70% | Google Font equivalents. Add `TikTokSans-Bold.ttf` for full coverage |
| Instagram story text | ~60% | Proprietary names, equivalent fonts included |

**Font databases:**

| File | Count | What it covers |
|------|-------|---------------|
| `fonts/tiktok.json` | 12 | TikTok native in-app font options + Google Font equivalents |
| `fonts/capcut.json` | 65 | CapCut font library |
| `fonts/canva_popular.json` | 110 | Most-used Canva fonts |
| `fonts/alias.json` | 30 | Maps platform-specific names to Google Font names |
| `fonts/ttf/` | 287 TTFs | Actual font files used for render-and-compare matching |

---

## Integrating into an agent

The cleanest integration is via the REST API. The agent sends a base64-encoded image and reads the JSON response.

**Python example:**
```python
import base64, requests

def identify_fonts(image_path: str, api_url: str = "http://localhost:8000") -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(f"{api_url}/detect/base64", json={"image": b64})
    return resp.json()

result = identify_fonts("slide.png")
for font in result["fonts_detected"]:
    print(f"{font['font']} {font['style']} — confidence {font['confidence']}")
    for inst in font["instances"]:
        print(f"  '{inst['text']}' at {inst['region']}, {inst['font_size_px']}px")
        print(f"  color={inst['text_color']}, outline={inst['has_outline']} ({inst['outline_color']})")
        print(f"  background={inst['background']} ({inst['background_color']})")
```

**Go example:**
```go
func IdentifyFonts(imageB64 string, apiURL string) (map[string]any, error) {
    body, _ := json.Marshal(map[string]string{"image": imageB64})
    resp, err := http.Post(apiURL+"/detect/base64", "application/json", bytes.NewReader(body))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    var result map[string]any
    json.NewDecoder(resp.Body).Decode(&result)
    return result, nil
}
```

**What to do with the result in Skybuds:**
- Use `font` + `style` to look up or recommend matching fonts
- Use `text_color` + `background_color` to extract the color palette of the slide
- Use `has_outline` + `outline_color` to replicate the exact text styling
- Use `region` + `font_size_px` to understand the layout hierarchy (headline vs subtext)
- Use `confidence` to decide whether to show a "best guess" label vs a confirmed match

---

## Confidence guide

| Confidence | Meaning | Agent behavior |
|-----------|---------|----------------|
| `>0.80` | High confidence, exact or near-exact match | Use the font name directly |
| `0.50–0.80` | Good estimate, likely the correct family | Use with a "best match" label |
| `0.30–0.50` | Best guess, font may not be in our database | Show top-3 candidates |
| `<0.30` | Very uncertain | Flag as unknown, return `top_fonts` array |

---

## Known limitations

- TikTok Sans (proprietary) requires manual TTF addition for high-confidence detection
- Heavily stylized text (glow, shadow, 3D effects) reduces confidence
- Text smaller than 10px is filtered out as unclassifiable
- Font weight variant (SemiBold vs Bold) may be off by one step; family name is reliable
- The tool detects creator-added text overlays. It filters TikTok/IG UI chrome (search bars, nav elements)
