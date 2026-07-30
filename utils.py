"""
utils.py — Supporting helpers for Text2Image Studio.

Keeps app.py focused on UI wiring by housing prompt composition, on-disk
generation history (JSON + PNG files), and small image utilities.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Optional

from PIL import Image

from config import STYLE_PRESETS, BASE_NEGATIVE_PROMPT, HISTORY_DIR

HISTORY_INDEX = os.path.join(HISTORY_DIR, "index.json")


def build_prompts(raw_prompt: str, raw_negative: str, style_key: str) -> tuple[str, str]:
    """Merge the user prompt with a style preset and the base negative prompt."""
    preset = STYLE_PRESETS.get(style_key, STYLE_PRESETS["None"])

    prompt = raw_prompt.strip()
    if preset.prompt_suffix:
        prompt = f"{prompt}, {preset.prompt_suffix}"

    negative_parts = [p for p in [raw_negative.strip(), preset.negative_suffix, BASE_NEGATIVE_PROMPT] if p]
    negative = ", ".join(negative_parts)

    return prompt, negative


def ensure_history_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_INDEX):
        with open(HISTORY_INDEX, "w") as f:
            json.dump([], f)


def save_to_history(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    model_key: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
) -> dict:
    """Persist an image + its generation metadata; returns the entry."""
    ensure_history_dir()

    entry_id = uuid.uuid4().hex[:12]
    filename = f"{entry_id}.png"
    filepath = os.path.join(HISTORY_DIR, filename)
    image.save(filepath)

    entry = {
        "id": entry_id,
        "filename": filename,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model_key": model_key,
        "seed": seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(HISTORY_INDEX, "r+") as f:
        data = json.load(f)
        data.insert(0, entry)  # newest first
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    return entry


def load_history(limit: Optional[int] = 50) -> list[dict]:
    ensure_history_dir()
    with open(HISTORY_INDEX, "r") as f:
        data = json.load(f)
    return data[:limit] if limit else data


def clear_history():
    ensure_history_dir()
    with open(HISTORY_INDEX, "r") as f:
        data = json.load(f)
    for entry in data:
        path = os.path.join(HISTORY_DIR, entry["filename"])
        if os.path.exists(path):
            os.remove(path)
    with open(HISTORY_INDEX, "w") as f:
        json.dump([], f)


def image_grid(images: list[Image.Image], cols: int = 2) -> Image.Image:
    """Compose a list of same-size images into a single contact-sheet grid."""
    if not images:
        raise ValueError("No images to compose into a grid.")
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(20, 20, 20))
    for idx, img in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid.paste(img, (x, y))
    return grid
