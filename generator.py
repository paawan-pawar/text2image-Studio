"""
generator.py — Model loading & inference engine for Text2Image Studio.

Wraps 🤗 diffusers pipelines behind a single, uniform interface so the UI
layer never needs to know which underlying architecture (SDXL / SDXL-Turbo /
FLUX) is currently active. Handles:

  * lazy, cached pipeline loading (swap models without re-downloading)
  * automatic device / dtype selection (CUDA fp16, MPS fp16, CPU fp32)
  * VRAM-friendly optimizations (attention slicing, VAE slicing/tiling,
    sequential CPU offload for large models like FLUX.1-dev)
  * LoRA weight loading
  * reproducible seeding
  * batch generation with progress callback
"""

from __future__ import annotations

import gc
import time
import random
from dataclasses import dataclass
from typing import Callable, Optional

import torch

try:
    import diffusers
except ImportError as exc:
    raise ImportError(
        "The diffusers library is required to run generator.py. "
        "Install dependencies with `pip install -r requirements.txt` "
        "or ensure the workspace venv is activated."
    ) from exc

from config import MODELS, SCHEDULERS, ModelSpec


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    """Choose the best available accelerator and matching precision."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str
    model_key: str
    steps: int
    guidance_scale: float
    width: int
    height: int
    seed: int
    num_images: int = 1
    scheduler_name: str = "Euler a"
    lora_path: Optional[str] = None
    lora_scale: float = 0.8
    init_image: Optional["object"] = None   # PIL.Image, for img2img
    strength: float = 0.6                    # img2img denoise strength


@dataclass
class GenerationResult:
    images: list
    seeds_used: list[int]
    elapsed_seconds: float
    model_key: str


class GenerationEngine:
    """Holds at most one loaded pipeline at a time to conserve VRAM."""

    def __init__(self):
        self.device, self.dtype = pick_device_and_dtype()
        self._loaded_key: Optional[str] = None
        self._pipe = None
        self._img2img_pipe = None

    # ------------------------------------------------------------------
    # Pipeline (model) management
    # ------------------------------------------------------------------
    def unload(self):
        self._pipe = None
        self._img2img_pipe = None
        self._loaded_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, model_key: str, progress_cb: Optional[Callable[[str], None]] = None):
        """Load (or reuse) the pipeline for the given model key."""
        if self._loaded_key == model_key and self._pipe is not None:
            return  # already loaded

        spec = MODELS[model_key]
        if progress_cb:
            progress_cb(f"Loading {spec.display_name} ({spec.repo_id}) …")

        # Free whatever was loaded before switching models.
        self.unload()

        import diffusers

        pipe_cls = getattr(diffusers, spec.pipeline_cls)
        pipe = pipe_cls.from_pretrained(
            spec.repo_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None,
        )

        pipe = self._apply_memory_optimizations(pipe, spec)
        self._pipe = pipe
        self._loaded_key = model_key

        if progress_cb:
            progress_cb(f"{spec.display_name} ready on {self.device}.")

    def _apply_memory_optimizations(self, pipe, spec: ModelSpec):
        """Trade a little speed for the ability to run on modest GPUs."""
        if self.device == "cuda":
            # Large / non-distilled models (e.g. FLUX.1-dev) benefit from
            # offloading submodules to CPU between steps instead of keeping
            # everything resident on the GPU.
            if spec.vram_gb_recommended >= 20:
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.to(self.device)
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pass  # full on-GPU is fine for <=16GB-class models
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
        else:
            pipe.to(self.device)

        return pipe

    def load_lora(self, lora_path: str, scale: float = 0.8):
        if self._pipe is None:
            raise RuntimeError("Load a base model before applying a LoRA.")
        self._pipe.load_lora_weights(lora_path)
        if hasattr(self._pipe, "fuse_lora"):
            self._pipe.fuse_lora(lora_scale=scale)

    def set_scheduler(self, scheduler_name: str):
        if self._pipe is None:
            return
        import diffusers

        scheduler_cls_name = SCHEDULERS.get(scheduler_name)
        if scheduler_cls_name and hasattr(diffusers, scheduler_cls_name):
            scheduler_cls = getattr(diffusers, scheduler_cls_name)
            self._pipe.scheduler = scheduler_cls.from_config(self._pipe.scheduler.config)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def generate(
        self,
        req: GenerationRequest,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> GenerationResult:
        spec = MODELS[req.model_key]
        self.load(req.model_key)
        if req.scheduler_name and not spec.is_distilled_fast:
            self.set_scheduler(req.scheduler_name)
        if req.lora_path:
            self.load_lora(req.lora_path, req.lora_scale)

        images = []
        seeds_used = []
        start = time.time()

        for i in range(req.num_images):
            seed = req.seed if req.seed is not None and req.seed >= 0 else random.randint(0, 2**32 - 1)
            # Vary seed per image in a batch unless user pinned an explicit seed and num_images == 1
            if req.num_images > 1 and req.seed is not None and req.seed >= 0:
                seed = req.seed + i

            generator = torch.Generator(device="cpu").manual_seed(seed)

            call_kwargs = dict(
                prompt=req.prompt,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                generator=generator,
            )
            if spec.supports_negative_prompt and not spec.is_distilled_fast:
                call_kwargs["negative_prompt"] = req.negative_prompt

            if req.init_image is not None:
                call_kwargs["image"] = req.init_image
                call_kwargs["strength"] = req.strength

            result = self._pipe(**call_kwargs)
            images.append(result.images[0])
            seeds_used.append(seed)

            if progress_cb:
                progress_cb(i + 1, req.num_images)

        elapsed = time.time() - start
        return GenerationResult(
            images=images,
            seeds_used=seeds_used,
            elapsed_seconds=elapsed,
            model_key=req.model_key,
        )
