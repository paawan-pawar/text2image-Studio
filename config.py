"""
config.py — Central configuration for Text2Image Studio.

Holds model registry (latest open text-to-image models supported via
🤗 diffusers) and curated style presets for prompt enhancement.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    key: str
    display_name: str
    repo_id: str
    pipeline_cls: str          # name of the diffusers pipeline class to use
    default_steps: int
    default_guidance: float
    max_resolution: int
    supports_negative_prompt: bool = True
    is_distilled_fast: bool = False   # turbo/schnell-style few-step models
    notes: str = ""
    vram_gb_recommended: int = 8


# ---------------------------------------------------------------------------
# Model registry — latest-generation open-weight text-to-image models
# ---------------------------------------------------------------------------
MODELS: dict[str, ModelSpec] = {
    "sdxl-base": ModelSpec(
        key="sdxl-base",
        display_name="Stable Diffusion XL 1.0 (Base + Refiner)",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_cls="StableDiffusionXLPipeline",
        default_steps=40,
        default_guidance=7.5,
        max_resolution=1024,
        is_distilled_fast=False,
        notes="High quality, general purpose. Supports an optional refiner pass.",
        vram_gb_recommended=10,
    ),
    "sdxl-turbo": ModelSpec(
        key="sdxl-turbo",
        display_name="SDXL Turbo (1-4 step, real-time)",
        repo_id="stabilityai/sdxl-turbo",
        pipeline_cls="AutoPipelineForText2Image",
        default_steps=2,
        default_guidance=0.0,  # turbo models are trained for CFG-free sampling
        max_resolution=1024,
        is_distilled_fast=True,
        notes="Adversarial-distilled SDXL. Near-instant previews, slightly lower fidelity.",
        vram_gb_recommended=8,
    ),
    "flux-schnell": ModelSpec(
        key="flux-schnell",
        display_name="FLUX.1 [schnell] (fast, Apache 2.0)",
        repo_id="black-forest-labs/FLUX.1-schnell",
        pipeline_cls="FluxPipeline",
        default_steps=4,
        default_guidance=0.0,
        max_resolution=1024,
        is_distilled_fast=True,
        notes="12B param rectified-flow transformer, distilled for 1-4 step inference.",
        vram_gb_recommended=16,
    ),
    "flux-dev": ModelSpec(
        key="flux-dev",
        display_name="FLUX.1 [dev] (highest fidelity, non-commercial)",
        repo_id="black-forest-labs/FLUX.1-dev",
        pipeline_cls="FluxPipeline",
        default_steps=28,
        default_guidance=3.5,
        max_resolution=1024,
        is_distilled_fast=False,
        notes="Best-in-class open-weight prompt adherence and detail. Needs more VRAM.",
        vram_gb_recommended=24,
    ),
}

DEFAULT_MODEL_KEY = "sdxl-turbo"

# ---------------------------------------------------------------------------
# Scheduler / sampler choices exposed in the UI (diffusers scheduler names)
# ---------------------------------------------------------------------------
SCHEDULERS = {
    "Euler a": "EulerAncestralDiscreteScheduler",
    "Euler": "EulerDiscreteScheduler",
    "DPM++ 2M Karras": "DPMSolverMultistepScheduler",
    "DDIM": "DDIMScheduler",
    "UniPC": "UniPCMultistepScheduler",
}

# ---------------------------------------------------------------------------
# Style presets — appended to the user prompt / negative prompt
# ---------------------------------------------------------------------------
@dataclass
class StylePreset:
    name: str
    prompt_suffix: str
    negative_suffix: str = ""


STYLE_PRESETS: dict[str, StylePreset] = {
    "None": StylePreset("None", "", ""),
    "Cinematic": StylePreset(
        "Cinematic",
        "cinematic lighting, dramatic composition, film grain, 35mm, shot on ARRI Alexa, ultra detailed",
        "flat lighting, cartoon, low detail",
    ),
    "Photorealistic": StylePreset(
        "Photorealistic",
        "photorealistic, ultra detailed, natural lighting, high dynamic range, DSLR photo, 85mm lens",
        "cartoon, illustration, painting, drawing, anime, unrealistic",
    ),
    "Anime": StylePreset(
        "Anime",
        "anime style, vibrant colors, clean line art, studio quality, detailed shading",
        "photorealistic, 3d render, western comic",
    ),
    "Digital Art": StylePreset(
        "Digital Art",
        "digital painting, trending on artstation, intricate detail, concept art, dramatic lighting",
        "blurry, low quality, watermark",
    ),
    "3D Render": StylePreset(
        "3D Render",
        "octane render, 3d, unreal engine 5, physically based rendering, subsurface scattering",
        "flat, 2d, sketch",
    ),
    "Minimalist": StylePreset(
        "Minimalist",
        "minimalist, clean composition, simple shapes, negative space, flat colors",
        "cluttered, busy, ornate, noisy",
    ),
}

# Global default negative prompt (quality guard) — always merged in
BASE_NEGATIVE_PROMPT = (
    "lowres, jpeg artifacts, blurry, deformed, disfigured, extra limbs, "
    "bad anatomy, watermark, signature, text, oversaturated"
)

# App-wide defaults
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
MAX_BATCH_SIZE = 4
HISTORY_DIR = "history"
