"""
Text2Image Studio — a Streamlit front-end over 🤗 diffusers text-to-image
pipelines (SDXL, SDXL-Turbo, FLUX.1-schnell, FLUX.1-dev).

Run with:  streamlit run app.py
"""

import io
import time

import streamlit as st
from PIL import Image

from config import (
    MODELS,
    DEFAULT_MODEL_KEY,
    SCHEDULERS,
    STYLE_PRESETS,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    MAX_BATCH_SIZE,
)
from generator import GenerationEngine, GenerationRequest
from utils import build_prompts, save_to_history, load_history, clear_history, image_grid

st.set_page_config(
    page_title="Text2Image Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
if "engine" not in st.session_state:
    st.session_state.engine = GenerationEngine()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "status_msg" not in st.session_state:
    st.session_state.status_msg = ""

engine: GenerationEngine = st.session_state.engine

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at top left, #1a1a2e 0%, #0f0f1a 60%); }
    h1, h2, h3 { color: #f5f5f7; }
    .stButton>button {
        background: linear-gradient(90deg, #7b2ff7, #f107a3);
        color: white; border: none; border-radius: 8px; font-weight: 600;
        padding: 0.6em 1.2em;
    }
    .stButton>button:hover { opacity: 0.9; }
    .model-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        background: #7b2ff7; color: white; font-size: 0.75em; margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎨 Text2Image Studio")
st.caption("Prompt-to-pixels playground built on the latest open diffusion models — SDXL, SDXL-Turbo & FLUX.1")

# ---------------------------------------------------------------------------
# Sidebar — model & generation settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model")

    model_key = st.selectbox(
        "Architecture",
        options=list(MODELS.keys()),
        format_func=lambda k: MODELS[k].display_name,
        index=list(MODELS.keys()).index(DEFAULT_MODEL_KEY),
    )
    spec = MODELS[model_key]
    st.caption(spec.notes)
    st.caption(f"Recommended VRAM: ~{spec.vram_gb_recommended} GB · Device detected: **{engine.device}**")

    st.divider()
    st.header("🎛️ Generation settings")

    steps = st.slider(
        "Inference steps", 1, 60, spec.default_steps,
        help="Distilled models (Turbo / schnell) only need 1-4 steps.",
    )
    guidance = st.slider(
        "Guidance scale (CFG)", 0.0, 15.0, spec.default_guidance, step=0.5,
        help="How strongly the image follows the prompt. Turbo/schnell models are tuned for 0.",
    )
    scheduler_name = st.selectbox(
        "Scheduler / sampler", list(SCHEDULERS.keys()),
        disabled=spec.is_distilled_fast,
    )

    col_w, col_h = st.columns(2)
    with col_w:
        width = st.select_slider("Width", options=[512, 768, 1024], value=DEFAULT_WIDTH)
    with col_h:
        height = st.select_slider("Height", options=[512, 768, 1024], value=DEFAULT_HEIGHT)

    seed = st.number_input(
        "Seed (-1 = random)", min_value=-1, max_value=2**32 - 1, value=-1,
        help="Pin a seed for reproducible results, or -1 for a fresh one each time.",
    )
    num_images = st.slider("Batch size", 1, MAX_BATCH_SIZE, 1)

    st.divider()
    st.header("🧩 LoRA (optional)")
    lora_path = st.text_input("LoRA repo id or local path", value="")
    lora_scale = st.slider("LoRA strength", 0.0, 1.5, 0.8, step=0.05, disabled=not lora_path)

    st.divider()
    st.header("🖼️ Image-to-image (optional)")
    init_image_file = st.file_uploader("Starting image", type=["png", "jpg", "jpeg"])
    strength = st.slider(
        "Denoise strength", 0.1, 1.0, 0.6, step=0.05,
        help="Lower = closer to the source image, higher = closer to a fresh generation.",
        disabled=init_image_file is None,
    )

# ---------------------------------------------------------------------------
# Main panel — prompt & output
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Prompt")
    prompt = st.text_area(
        "Describe the image you want",
        placeholder="A lighthouse on a cliff at golden hour, dramatic waves, ultra detailed",
        height=110,
    )
    negative_prompt = st.text_area(
        "Negative prompt (what to avoid)",
        placeholder="extra fingers, blurry, low quality",
        height=70,
    )
    style_key = st.selectbox("Style preset", list(STYLE_PRESETS.keys()))

    generate_clicked = st.button("✨ Generate", use_container_width=True, type="primary")

    status_placeholder = st.empty()
    progress_bar = st.progress(0, text="Idle")

with right:
    st.subheader("Output")
    output_placeholder = st.empty()
    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        if len(result.images) > 1:
            output_placeholder.image(image_grid(result.images), use_container_width=True)
        else:
            output_placeholder.image(result.images[0], use_container_width=True)
        st.caption(
            f"Model: {MODELS[result.model_key].display_name} · "
            f"Seeds: {result.seeds_used} · {result.elapsed_seconds:.1f}s"
        )
        for i, img in enumerate(result.images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button(
                f"Download image {i + 1}",
                data=buf.getvalue(),
                file_name=f"text2image_{result.seeds_used[i]}.png",
                mime="image/png",
                key=f"dl_{i}_{result.seeds_used[i]}",
            )

# ---------------------------------------------------------------------------
# Generation trigger
# ---------------------------------------------------------------------------
if generate_clicked:
    if not prompt.strip():
        st.warning("Enter a prompt first.")
    else:
        final_prompt, final_negative = build_prompts(prompt, negative_prompt, style_key)

        init_image = None
        if init_image_file is not None:
            init_image = Image.open(init_image_file).convert("RGB").resize((width, height))

        def on_load_progress(msg: str):
            status_placeholder.info(msg)

        def on_gen_progress(done: int, total: int):
            progress_bar.progress(done / total, text=f"Generating image {done}/{total}")

        try:
            engine.load(model_key, progress_cb=on_load_progress)
            req = GenerationRequest(
                prompt=final_prompt,
                negative_prompt=final_negative,
                model_key=model_key,
                steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                seed=seed,
                num_images=num_images,
                scheduler_name=scheduler_name,
                lora_path=lora_path or None,
                lora_scale=lora_scale,
                init_image=init_image,
                strength=strength,
            )
            result = engine.generate(req, progress_cb=on_gen_progress)
            st.session_state.last_result = result

            for img, s in zip(result.images, result.seeds_used):
                save_to_history(
                    img, final_prompt, final_negative, model_key, s,
                    steps, guidance, width, height,
                )

            status_placeholder.success(f"Done in {result.elapsed_seconds:.1f}s")
            st.rerun()

        except Exception as e:
            status_placeholder.error(f"Generation failed: {e}")

# ---------------------------------------------------------------------------
# History gallery
# ---------------------------------------------------------------------------
st.divider()
hist_col1, hist_col2 = st.columns([4, 1])
with hist_col1:
    st.subheader("🕘 History")
with hist_col2:
    if st.button("Clear history", use_container_width=True):
        clear_history()
        st.rerun()

entries = load_history(limit=12)
if not entries:
    st.caption("No generations yet — your history will appear here.")
else:
    cols = st.columns(4)
    for i, entry in enumerate(entries):
        with cols[i % 4]:
            img_path = f"history/{entry['filename']}"
            st.image(img_path, use_container_width=True)
            st.caption(f"{entry['prompt'][:60]}…" if len(entry["prompt"]) > 60 else entry["prompt"])
            st.caption(f"seed {entry['seed']} · {MODELS.get(entry['model_key'], entry['model_key']).display_name if entry['model_key'] in MODELS else entry['model_key']}")
