# 🎨 Text2Image Studio

A full-featured text-to-image generation app built on the latest open
diffusion models, wrapped in a polished Streamlit UI. Built as a
Generative AI portfolio project.
Deploy link: https://text2image-studio-hqfiycjttdbebvmm7cjfaj.streamlit.app/

## What's inside

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — prompt input, settings, gallery, history |
| `generator.py` | Model-agnostic inference engine (loading, memory optimization, batching) |
| `config.py` | Model registry, scheduler options, style presets |
| `utils.py` | Prompt composition, on-disk history (JSON + PNG) |
| `requirements.txt` | Pinned dependencies |

## Supported models (swap from the sidebar, no code changes needed)

- **Stable Diffusion XL 1.0** — high quality, general purpose
- **SDXL Turbo** — 1–4 step near-real-time generation
- **FLUX.1 [schnell]** — Apache-2.0, distilled, very fast, excellent prompt adherence
- **FLUX.1 [dev]** — best open-weight fidelity available today (non-commercial license)

Adding a new model later is a one-line addition to `MODELS` in `config.py`.

## Features

- Model switching (SDXL / SDXL-Turbo / FLUX schnell / FLUX dev)
- Style presets (Cinematic, Photorealistic, Anime, Digital Art, 3D Render, Minimalist)
- Negative prompts + a built-in quality-guard negative prompt
- Scheduler / sampler selection (Euler a, Euler, DPM++ 2M Karras, DDIM, UniPC)
- Adjustable steps, CFG guidance scale, resolution (512/768/1024), batch size (up to 4)
- Reproducible generation via seed pinning
- Image-to-image (upload a starting image + denoise strength)
- LoRA loading (any HF repo id or local `.safetensors` path)
- Automatic memory optimization: attention/VAE slicing, sequential CPU
  offload for large models (FLUX.1-dev) so it can run on smaller GPUs
- Persistent history gallery with download buttons

## Running it

### Requirements
- Python 3.10+
- A CUDA GPU is strongly recommended. VRAM guide:
  - SDXL Turbo / FLUX schnell: 8–16 GB
  - SDXL base: ~10 GB
  - FLUX.1-dev: 24 GB (or less, with CPU offload enabled — slower)
- CPU-only will work but generation will take minutes per image.

### Setup

```bash
git clone <your-repo-url> text2image-studio
cd text2image-studio
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Log in once so diffusers can pull gated/rate-limited weights
huggingface-cli login

streamlit run app.py
```

The first generation with a given model will download its weights
(several GB) from Hugging Face Hub and cache them locally — subsequent
runs are instant.

### Running on Google Colab

1. Open a new Colab notebook, set **Runtime → Change runtime type → T4/A100 GPU**.
2. ```python
   !pip install -q streamlit diffusers transformers accelerate safetensors peft sentencepiece
   !npm install -g localtunnel
   ```
3. Upload `app.py`, `generator.py`, `config.py`, `utils.py` to the Colab file browser (or `git clone` your repo).
4. ```python
   !streamlit run app.py &>/content/logs.txt &
   !npx localtunnel --port 8501
   ```
5. Open the printed `localtunnel` URL — it will ask for the tunnel
   "password", which is the public IP printed just above the link.

### FLUX.1 access note
FLUX.1 [dev] and [schnell] weights are hosted on Hugging Face and may
require accepting the model license on the model page once, and being
logged in via `huggingface-cli login`, before the first download.

## Architecture notes (why it's built this way)

- **One pipeline resident at a time.** `GenerationEngine` unloads the
  previous model before loading a new one, so you can flip between
  SDXL and FLUX without needing 2× the VRAM.
- **Model-agnostic `GenerationRequest`.** The UI never imports a
  diffusers pipeline class directly — it only builds a request object,
  so adding a new architecture is isolated to `config.py` +
  `generator.py`.
- **Distilled models skip CFG/negative prompts.** SDXL Turbo and FLUX
  schnell are trained for guidance-free, few-step sampling; the engine
  automatically omits `negative_prompt`/high CFG for those to match
  how they were trained.
- **History is just files.** No database — a JSON index plus PNGs on
  disk, so the gallery survives app restarts and is trivial to inspect
  or back up.

## Extending it

- Add ControlNet (pose/depth/canny conditioning) by adding a
  `pipeline_cls: "StableDiffusionXLControlNetPipeline"` entry and
  passing a control image through `GenerationRequest`.
- Add prompt auto-enhancement by piping the raw prompt through a small
  local LLM (or the Anthropic API) before `build_prompts()`.
- Swap the JSON+PNG history for a proper database if you need
  multi-user support.
