# 🍌 ComfyUI-Gemini-NanoBanana

Custom nodes for **Google Gemini Nano Banana** image generation and editing in ComfyUI.

Supports both **Nano Banana** (Gemini 2.5 Flash Image) for fast generation and **Nano Banana Pro** (Gemini 3 Pro Image) for professional 4K output with advanced reasoning.

## Features

- **Text → Image** (all-in-one) — API key, model dropdown, and generation in a single node
- **Image → Image** — Edit/transform images with text instructions (standalone or chained)
- **Multi-Image Compose** — Blend up to 4 input images into one
- **Chat Session** — Multi-turn conversational editing with history chaining
- **Describe Image** — Vision analysis / captioning
- **Save Image** — PNG/JPEG/WEBP output with quality control
- **Auto Model Discovery** — Model dropdown auto-populates from the API at startup
- **Custom Model Override** — Type any model name manually for new/unlisted models
- **Google Search Grounding** — Generate images based on real-time data
- **Resolution Control** — 1K, 2K, 4K output + flexible aspect ratios
- **Flexible API Key** — Enter directly in node, use `.env` file, or environment variable

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Gerlif/comfy-banana-kasger.git
cd comfy-banana-kasger
pip install -r requirements.txt
```

## API Key Setup

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

Three options (in priority order):

1. **In the node** — Enter the key in the `api_key` field
2. **`.env` file** — Copy `.env.example` to `.env` and paste your key
3. **Environment variable** — `export GEMINI_API_KEY=your_key_here`

> **Tip:** If you set the env var or `.env` file, the model dropdown auto-populates from the API when ComfyUI starts.

## Nodes Overview

### 🍌 Nano Banana Text → Image

The primary all-in-one node. Handles API config, model selection, and image generation.

| Input | Type | Description |
|-------|------|-------------|
| `prompt` | STRING | Text description (required) |
| `model` | COMBO | Auto-populated model dropdown (required) |
| `api_key` | STRING | API key (optional — falls back to env/.env) |
| `custom_model` | STRING | Manual model name override |
| `aspect_ratio` | COMBO | auto, 1:1, 3:4, 4:3, 9:16, 16:9, 21:9 |
| `image_size` | COMBO | auto, 1K, 2K, 4K |
| `temperature` | FLOAT | Creativity (0.0–2.0) |
| `use_google_search` | BOOLEAN | Ground generation in real-time data |
| `system_instruction` | STRING | Optional system prompt |
| `seed` | INT | Advisory seed |

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Generated image(s) |
| `response_text` | STRING | Model's text response |
| `gemini_config` | GEMINI_CONFIG | Pass to other Nano Banana nodes |

### 🍌 Nano Banana Image → Image

Edit/transform images. Works **standalone** (own API key + model inputs) or chained via `gemini_config`.

### 🍌 Nano Banana Multi-Image Compose

Blend up to 4 images with a text prompt. Requires `gemini_config` input.

### 🍌 Nano Banana Chat Session

Multi-turn editing — chain multiple Chat Session nodes. Pass `chat_history` output → next node's `chat_history` input. Also passes `gemini_config` through.

### 🍌 Gemini Describe Image

Vision/analysis — returns text description only. Requires `gemini_config`.

### 🍌 Gemini Save Image

Save output images in PNG/JPEG/WEBP with configurable quality.

## Example Workflows

### Basic Text to Image
```
[🍌 Text → Image] → [Preview Image]
```

### Image Editing (standalone)
```
[Load Image] → [🍌 Image → Image] → [Preview Image]
```

### Image Editing (chained config)
```
[🍌 Text → Image] ──image──→ [Preview Image]
        │
        └─gemini_config──→ [🍌 Image → Image] → [Preview Image]
```

### Multi-Turn Editing
```
[🍌 Text → Image] ─config─→ [🍌 Chat: "Draw a house"] ─chat_history─→ [🍌 Chat: "Add a garden"] → [Preview]
```

## Supported Models

The dropdown auto-discovers models from the API. Known image models:

| Model | Description |
|-------|-------------|
| `gemini-2.5-flash-image` | **Nano Banana** — Fast, efficient |
| `gemini-3-pro-image-preview` | **Nano Banana Pro** — 4K, advanced reasoning, text rendering |

Use the `custom_model` field to specify any new model not yet in the dropdown.

## License

MIT
