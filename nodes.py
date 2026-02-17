"""
ComfyUI Custom Nodes for Gemini Nano Banana Image Generation
Supports text-to-image, image-to-image editing, and multi-turn conversations
using Google's Gemini API (Nano Banana / Nano Banana Pro).
"""

import os
import io
import json
import base64
import time
import requests
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENV_FILE_PATHS = [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(__file__), "../../.env"),
]


def _load_env_api_key() -> str:
    """Try to load GEMINI_API_KEY from .env files or environment."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    for p in ENV_FILE_PATHS:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def _resolve_api_key(node_key: str) -> str:
    """Return the API key from the node input, falling back to env/.env."""
    if node_key and node_key.strip() and node_key.strip() != "":
        return node_key.strip()
    key = _load_env_api_key()
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Either enter it in the node, "
            "set the GEMINI_API_KEY environment variable, or create a .env file "
            "in the custom node folder."
        )
    return key


API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI IMAGE tensor [B,H,W,C] float32 0-1."""
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor [B,H,W,C] to PIL (first image in batch)."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, tuple[list[str], float]] = {}
_CACHE_TTL = 300  # 5 minutes


def _default_models() -> list[str]:
    """Fallback model list if API call fails."""
    return [
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-preview-image",
        "gemini-3-pro-image-preview",
    ]


def _fetch_image_models(api_key: str, force: bool = False) -> list[str]:
    """Fetch model list from Gemini API, filtering for image-capable models."""
    now = time.time()
    if not force and api_key in _MODEL_CACHE:
        cached_models, cached_time = _MODEL_CACHE[api_key]
        if now - cached_time < _CACHE_TTL:
            return cached_models

    url = f"{API_BASE}/models?key={api_key}&pageSize=1000"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[Gemini NanoBanana] Failed to fetch models: {e}")
        return _default_models()

    image_models = []
    for m in data.get("models", []):
        name = m.get("name", "")
        short = name.replace("models/", "")
        supported = m.get("supportedGenerationMethods", [])
        desc = m.get("description", "").lower()
        display = m.get("displayName", "").lower()
        if any(kw in short for kw in ["image", "nano-banana", "nano_banana"]):
            image_models.append(short)
        elif "image" in desc and "generateContent" in supported:
            image_models.append(short)
        elif "nano banana" in display:
            image_models.append(short)

    if not image_models:
        return _default_models()

    result = sorted(set(image_models))
    _MODEL_CACHE[api_key] = (result, now)
    return result


def _get_model_choices(api_key: str) -> list[str]:
    """Get model list, using cache when available."""
    return _fetch_image_models(api_key)


def _get_startup_models() -> list[str]:
    """Attempt to populate model dropdown at ComfyUI startup."""
    try:
        env_key = _load_env_api_key()
        if env_key:
            return _get_model_choices(env_key)
    except Exception:
        pass
    return _default_models()


# Pre-fetch at import time so the dropdown is populated
_STARTUP_MODELS = _get_startup_models()


# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------

def _generate_content(
    api_key: str,
    model: str,
    contents: list[dict],
    aspect_ratio: str | None = None,
    image_size: str | None = None,
    temperature: float | None = None,
    use_google_search: bool = False,
) -> dict:
    """Call Gemini generateContent endpoint."""
    url = f"{API_BASE}/models/{model}:generateContent?key={api_key}"

    generation_config: dict = {
        "responseModalities": ["TEXT", "IMAGE"],
    }
    image_config = {}
    if aspect_ratio and aspect_ratio != "auto":
        image_config["aspectRatio"] = aspect_ratio
    if image_size and image_size != "auto":
        image_config["imageSize"] = image_size
    if image_config:
        generation_config["imageConfig"] = image_config
    if temperature is not None:
        generation_config["temperature"] = temperature

    body: dict = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    if use_google_search:
        body["tools"] = [{"googleSearch": {}}]

    resp = requests.post(url, json=body, timeout=120)
    if resp.status_code != 200:
        error_msg = resp.text
        try:
            error_data = resp.json()
            error_msg = error_data.get("error", {}).get("message", resp.text)
        except Exception:
            pass
        raise RuntimeError(f"Gemini API error ({resp.status_code}): {error_msg}")

    return resp.json()


def _parse_response(data: dict) -> tuple[list[Image.Image], str]:
    """Parse generateContent response into images and text."""
    images: list[Image.Image] = []
    texts: list[str] = []

    candidates = data.get("candidates", [])
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            if "text" in part:
                texts.append(part["text"])
            elif "inlineData" in part:
                inline = part["inlineData"]
                img_bytes = base64.b64decode(inline["data"])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)

    return images, "\n".join(texts)


def _make_blank_result(msg: str = "") -> tuple[torch.Tensor, str]:
    blank = Image.new("RGB", (512, 512), (0, 0, 0))
    text = msg or "[No image generated. Try adjusting your prompt.]"
    return (_pil_to_tensor(blank), text)


def _images_to_batch(images: list[Image.Image]) -> torch.Tensor:
    tensors = [_pil_to_tensor(img) for img in images]
    return torch.cat(tensors, dim=0)


# ---------------------------------------------------------------------------
# Node: Nano Banana Text to Image  (all-in-one: config + fetch + generate)
# ---------------------------------------------------------------------------

class GeminiTextToImage:
    """All-in-one text-to-image node with built-in API config and model discovery.
    Also outputs gemini_config for chaining to other Nano Banana nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "A beautiful sunset over the ocean",
                    "multiline": True,
                }),
                "model": (_STARTUP_MODELS, {
                    "default": _STARTUP_MODELS[0] if _STARTUP_MODELS else "gemini-2.5-flash-image",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Gemini API key. Leave empty to use GEMINI_API_KEY env var or .env file.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Override model dropdown. Type a model name manually (e.g. a newly released model not yet in the dropdown).",
                }),
                "aspect_ratio": (["auto", "1:1", "3:4", "4:3", "9:16", "16:9", "21:9"],
                                 {"default": "auto"}),
                "image_size": (["auto", "1K", "2K", "4K"], {"default": "auto"}),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "use_google_search": ("BOOLEAN", {"default": False}),
                "system_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional system instruction to guide the model's style or behavior.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Seed for reproducibility (advisory, not guaranteed by API).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "GEMINI_CONFIG",)
    RETURN_NAMES = ("image", "response_text", "gemini_config",)
    FUNCTION = "generate"
    CATEGORY = "Gemini/NanoBanana"
    DESCRIPTION = (
        "All-in-one Nano Banana text-to-image node.\n"
        "Handles API key, model selection (auto-fetched from API), and generation.\n"
        "The gemini_config output can be passed to other Nano Banana nodes."
    )

    def generate(
        self,
        prompt: str,
        model: str,
        api_key: str = "",
        custom_model: str = "",
        aspect_ratio: str = "auto",
        image_size: str = "auto",
        temperature: float = 1.0,
        use_google_search: bool = False,
        system_instruction: str = "",
        seed: int = 0,
    ):
        resolved_key = _resolve_api_key(api_key)
        chosen_model = custom_model.strip() if custom_model.strip() else model

        # Refresh model cache in background for next run
        _fetch_image_models(resolved_key)

        config = {"api_key": resolved_key, "model": chosen_model}

        # Build contents
        text_content = prompt
        if system_instruction.strip():
            text_content = f"[System instruction]: {system_instruction.strip()}\n\n{prompt}"

        contents = [{"role": "user", "parts": [{"text": text_content}]}]

        data = _generate_content(
            api_key=resolved_key,
            model=chosen_model,
            contents=contents,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            temperature=temperature,
            use_google_search=use_google_search,
        )

        images, text = _parse_response(data)

        if not images:
            batch, text = _make_blank_result(text)
            return (batch, text, config)

        return (_images_to_batch(images), text, config)


# ---------------------------------------------------------------------------
# Node: Image to Image (Edit) — also standalone-capable
# ---------------------------------------------------------------------------

class GeminiImageToImage:
    """Edit or transform images using Gemini Nano Banana with text instructions.
    Can run standalone (own API key + model) or receive gemini_config from Text→Image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "Make this image look like a watercolor painting",
                    "multiline": True,
                }),
                "model": (_STARTUP_MODELS, {
                    "default": _STARTUP_MODELS[0] if _STARTUP_MODELS else "gemini-2.5-flash-image",
                }),
            },
            "optional": {
                "gemini_config": ("GEMINI_CONFIG",),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "API key. Ignored if gemini_config is connected.",
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Override model dropdown with a manual model name.",
                }),
                "aspect_ratio": (["auto", "1:1", "3:4", "4:3", "9:16", "16:9", "21:9"],
                                 {"default": "auto"}),
                "image_size": (["auto", "1K", "2K", "4K"], {"default": "auto"}),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "use_google_search": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "GEMINI_CONFIG",)
    RETURN_NAMES = ("image", "response_text", "gemini_config",)
    FUNCTION = "edit"
    CATEGORY = "Gemini/NanoBanana"
    DESCRIPTION = (
        "Edit images with text instructions using Gemini Nano Banana.\n"
        "Works standalone or with gemini_config from Text→Image."
    )

    def edit(
        self,
        image: torch.Tensor,
        prompt: str,
        model: str = "gemini-2.5-flash-image",
        gemini_config: dict | None = None,
        api_key: str = "",
        custom_model: str = "",
        aspect_ratio: str = "auto",
        image_size: str = "auto",
        temperature: float = 1.0,
        use_google_search: bool = False,
        seed: int = 0,
    ):
        if gemini_config:
            resolved_key = gemini_config["api_key"]
            chosen_model = gemini_config["model"]
        else:
            resolved_key = _resolve_api_key(api_key)
            chosen_model = custom_model.strip() if custom_model.strip() else model

        config = {"api_key": resolved_key, "model": chosen_model}

        pil_img = _tensor_to_pil(image)
        img_b64 = _pil_to_base64(pil_img, "PNG")

        contents = [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/png", "data": img_b64}},
            ],
        }]

        data = _generate_content(
            api_key=resolved_key, model=chosen_model, contents=contents,
            aspect_ratio=aspect_ratio, image_size=image_size,
            temperature=temperature, use_google_search=use_google_search,
        )

        images, text = _parse_response(data)

        if not images:
            return (_pil_to_tensor(pil_img), text or "[No edited image returned.]", config)

        return (_images_to_batch(images), text, config)


# ---------------------------------------------------------------------------
# Node: Multi-Image Blend / Compose
# ---------------------------------------------------------------------------

class GeminiMultiImageCompose:
    """Combine multiple images with a text prompt using Gemini."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_config": ("GEMINI_CONFIG",),
                "prompt": ("STRING", {
                    "default": "Combine these images into a single cohesive scene",
                    "multiline": True,
                }),
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "aspect_ratio": (["auto", "1:1", "3:4", "4:3", "9:16", "16:9", "21:9"],
                                 {"default": "auto"}),
                "image_size": (["auto", "1K", "2K", "4K"], {"default": "auto"}),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "use_google_search": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "response_text",)
    FUNCTION = "compose"
    CATEGORY = "Gemini/NanoBanana"
    DESCRIPTION = "Combine up to 4 images with text instructions. Requires gemini_config."

    def compose(
        self,
        gemini_config: dict,
        prompt: str,
        image1: torch.Tensor,
        image2: torch.Tensor | None = None,
        image3: torch.Tensor | None = None,
        image4: torch.Tensor | None = None,
        aspect_ratio: str = "auto",
        image_size: str = "auto",
        temperature: float = 1.0,
        use_google_search: bool = False,
    ):
        api_key = gemini_config["api_key"]
        model_name = gemini_config["model"]

        parts: list[dict] = [{"text": prompt}]
        for img_tensor in [image1, image2, image3, image4]:
            if img_tensor is not None:
                pil_img = _tensor_to_pil(img_tensor)
                img_b64 = _pil_to_base64(pil_img, "PNG")
                parts.append({"inlineData": {"mimeType": "image/png", "data": img_b64}})

        contents = [{"role": "user", "parts": parts}]

        data = _generate_content(
            api_key=api_key, model=model_name, contents=contents,
            aspect_ratio=aspect_ratio, image_size=image_size,
            temperature=temperature, use_google_search=use_google_search,
        )

        images, text = _parse_response(data)
        if not images:
            batch, text = _make_blank_result(text)
            return (batch, text)

        return (_images_to_batch(images), text)


# ---------------------------------------------------------------------------
# Node: Multi-Turn Chat (Conversational Editing)
# ---------------------------------------------------------------------------

class GeminiChatSession:
    """Multi-turn conversation for iterative image editing.
    Chain multiple Chat Session nodes together via chat_history."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_config": ("GEMINI_CONFIG",),
                "prompt": ("STRING", {
                    "default": "Create a clean whiteboard",
                    "multiline": True,
                }),
            },
            "optional": {
                "chat_history": ("GEMINI_CHAT",),
                "input_image": ("IMAGE",),
                "aspect_ratio": (["auto", "1:1", "3:4", "4:3", "9:16", "16:9", "21:9"],
                                 {"default": "auto"}),
                "image_size": (["auto", "1K", "2K", "4K"], {"default": "auto"}),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "use_google_search": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "GEMINI_CHAT", "GEMINI_CONFIG",)
    RETURN_NAMES = ("image", "response_text", "chat_history", "gemini_config",)
    FUNCTION = "chat"
    CATEGORY = "Gemini/NanoBanana"
    DESCRIPTION = (
        "Multi-turn conversational image editing.\n"
        "Chain multiple Chat Session nodes for iterative refinement.\n"
        "Pass chat_history from one to the next."
    )

    def chat(
        self,
        gemini_config: dict,
        prompt: str,
        chat_history: list | None = None,
        input_image: torch.Tensor | None = None,
        aspect_ratio: str = "auto",
        image_size: str = "auto",
        temperature: float = 1.0,
        use_google_search: bool = False,
    ):
        api_key = gemini_config["api_key"]
        model_name = gemini_config["model"]

        history = list(chat_history) if chat_history else []

        # Build user turn
        parts: list[dict] = [{"text": prompt}]
        if input_image is not None:
            pil_img = _tensor_to_pil(input_image)
            img_b64 = _pil_to_base64(pil_img, "PNG")
            parts.append({"inlineData": {"mimeType": "image/png", "data": img_b64}})

        history.append({"role": "user", "parts": parts})

        data = _generate_content(
            api_key=api_key, model=model_name, contents=history,
            aspect_ratio=aspect_ratio, image_size=image_size,
            temperature=temperature, use_google_search=use_google_search,
        )

        images, text = _parse_response(data)

        # Store full model response in history for multi-turn continuity
        candidates = data.get("candidates", [])
        for candidate in candidates:
            model_content = candidate.get("content", {})
            if model_content.get("parts"):
                history.append({"role": "model", "parts": model_content["parts"]})
                break

        if not images:
            batch, text = _make_blank_result(text)
            return (batch, text, history, gemini_config)

        return (_images_to_batch(images), text, history, gemini_config)


# ---------------------------------------------------------------------------
# Node: Describe Image (Vision)
# ---------------------------------------------------------------------------

class GeminiDescribeImage:
    """Use Gemini to describe or analyze an image. Text-only output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_config": ("GEMINI_CONFIG",),
                "image": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "describe"
    CATEGORY = "Gemini/NanoBanana"
    DESCRIPTION = "Analyze or describe an image using Gemini vision."

    def describe(
        self,
        gemini_config: dict,
        image: torch.Tensor,
        prompt: str = "Describe this image in detail.",
        temperature: float = 0.7,
    ):
        api_key = gemini_config["api_key"]
        model_name = gemini_config["model"]

        pil_img = _tensor_to_pil(image)
        img_b64 = _pil_to_base64(pil_img, "PNG")

        contents = [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/png", "data": img_b64}},
            ],
        }]

        # Text-only response
        url = f"{API_BASE}/models/{model_name}:generateContent?key={api_key}"
        body = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }

        resp = requests.post(url, json=body, timeout=60)
        if resp.status_code != 200:
            error_msg = resp.text
            try:
                error_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                pass
            raise RuntimeError(f"Gemini API error ({resp.status_code}): {error_msg}")

        data = resp.json()
        texts = []
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    texts.append(part["text"])

        return ("\n".join(texts) if texts else "[No description returned.]",)


# ---------------------------------------------------------------------------
# Node: Save Gemini Image
# ---------------------------------------------------------------------------

class GeminiSaveImage:
    """Save generated images to disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "gemini_nanobanana"}),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Leave empty for ComfyUI default output directory.",
                }),
                "format": (["PNG", "JPEG", "WEBP"], {"default": "PNG"}),
                "quality": ("INT", {
                    "default": 95, "min": 1, "max": 100,
                    "tooltip": "JPEG/WEBP quality (1-100).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_paths",)
    FUNCTION = "save"
    CATEGORY = "Gemini/NanoBanana"
    OUTPUT_NODE = True
    DESCRIPTION = "Save Gemini-generated images to disk."

    def save(
        self,
        image: torch.Tensor,
        filename_prefix: str = "gemini_nanobanana",
        output_dir: str = "",
        format: str = "PNG",
        quality: int = 95,
    ):
        if not output_dir:
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
            except ImportError:
                output_dir = os.path.join(os.path.dirname(__file__), "../../output")

        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        batch_size = image.shape[0]

        for i in range(batch_size):
            pil_img = _tensor_to_pil(image[i].unsqueeze(0))

            counter = 0
            while True:
                ext = format.lower()
                if ext == "jpeg":
                    ext = "jpg"
                fname = f"{filename_prefix}_{counter:05d}.{ext}"
                fpath = os.path.join(output_dir, fname)
                if not os.path.exists(fpath):
                    break
                counter += 1

            save_kwargs = {}
            if format in ("JPEG", "WEBP"):
                save_kwargs["quality"] = quality
            if format == "PNG":
                save_kwargs["compress_level"] = 4

            pil_img.save(fpath, format=format, **save_kwargs)
            saved_paths.append(fpath)
            print(f"[Gemini NanoBanana] Saved: {fpath}")

        return ("\n".join(saved_paths),)


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GeminiTextToImage": GeminiTextToImage,
    "GeminiImageToImage": GeminiImageToImage,
    "GeminiMultiImageCompose": GeminiMultiImageCompose,
    "GeminiChatSession": GeminiChatSession,
    "GeminiDescribeImage": GeminiDescribeImage,
    "GeminiSaveImage": GeminiSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiTextToImage": "\U0001f34c Nano Banana Text \u2192 Image",
    "GeminiImageToImage": "\U0001f34c Nano Banana Image \u2192 Image",
    "GeminiMultiImageCompose": "\U0001f34c Nano Banana Multi-Image Compose",
    "GeminiChatSession": "\U0001f34c Nano Banana Chat Session",
    "GeminiDescribeImage": "\U0001f34c Gemini Describe Image",
    "GeminiSaveImage": "\U0001f34c Gemini Save Image",
}
