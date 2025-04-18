import streamlit as st
from diffusers import StableDiffusionPipeline
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
from utils.image_generator import generate_image
from utils.music_generator import generate_music
from utils.ui_config import apply_custom_css
from ai_agent import ai_agent
from refiner_agent import refine_prompt

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI Generator", layout="wide")
apply_custom_css()

# ---- SESSION STATE INIT ----
for key in ["show_sidebar", "prompt", "prompt_history", "generated_image", "generated_music", "refined_prompt", "show_keyword_input", "refine_keywords"]:
    if key not in st.session_state:
        if key in ["show_sidebar", "show_keyword_input"]:
            st.session_state[key] = False
        elif key == "prompt_history":
            st.session_state[key] = []
        else:
            st.session_state[key] = ""

# ---- SIDEBAR TOGGLE ----
if st.button("\u2630 Menu", key="menu_button"):
    st.session_state.show_sidebar = not st.session_state.show_sidebar

if st.session_state.show_sidebar:
    with st.sidebar:
        st.header("History")
        if st.session_state.prompt_history:
            for i, p in enumerate(reversed(st.session_state.prompt_history), 1):
                st.write(f"{i}. {p}")
        else:
            st.write("No prompts yet.")

# ---- LOAD MODELS ----
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd_model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to(device)
    sd_model.enable_attention_slicing()
    music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to("cpu")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    return sd_model, music_model, processor, device

sd_model, music_model, processor, device = load_models()

# ---- MAIN UI ----
st.title("🎨 AI Image & 🎵 Music Generator")

# Input Prompt Area
prompt = st.text_area("Enter your prompt:", st.session_state.get("prompt", ""), height=150, key="prompt_input_box")

# Image Size
size = st.radio("Image Size:", ["Small", "Medium", "Large"], key="size_radio")
size_map = {"Small": (256, 256), "Medium": (384, 384), "Large": (512, 512)}
img_size = size_map[size]

# Description Type
st.session_state.description_type = st.selectbox("Description Length:", ["short", "medium", "long"], key="description_select")

# --- Refine Prompt ---
if st.button("🔁 Refine Prompt", key="refine_button_main"):
    st.session_state.show_keyword_input = not st.session_state.show_keyword_input

# Keyword Input Box
if st.session_state.show_keyword_input:
    st.session_state.refine_keywords = st.text_input("Enter keywords (comma-separated):", key="keyword_input_box")
    if st.button("✅ Generate Refined Prompt", key="generate_refined_prompt_button"):
        if not st.session_state.prompt or not st.session_state.prompt.strip():
            st.warning("Please write something in the main prompt before refining.")
        else:
            refined = refine_prompt(
                original_prompt=st.session_state.prompt,
                keywords=st.session_state.refine_keywords,
                max_length={"short": 30, "medium": 60, "long": 100}[st.session_state.description_type]
            )
            st.session_state.prompt = refined
            st.session_state.refined_prompt = refined
            st.session_state.show_keyword_input = False
            st.success("Prompt refined using your keywords!")
      
# Save prompt in history
if st.session_state.prompt and (
    len(st.session_state.prompt_history) == 0 or st.session_state.prompt_history[-1] != st.session_state.prompt
):
    st.session_state.prompt_history.append(st.session_state.prompt)

# --- GENERATE IMAGE ---
if st.button("🎨 Generate Image", key="generate_image_button"):
    if st.session_state.prompt:
        st.session_state.generated_image = generate_image(st.session_state.prompt, sd_model, img_size)
        st.success("Image generated successfully!")
    else:
        st.warning("Please enter a prompt!")

# --- GENERATE MUSIC ---
if st.button("🎵 Generate Music", key="generate_music_button"):
    if st.session_state.prompt:
        st.session_state.generated_music = generate_music(st.session_state.prompt, music_model, processor)
        st.success("Music generated successfully!")
    else:
        st.warning("Please enter a prompt!")

# --- AI AGENT BUTTON ---
if st.button("🤖 Let AI Agent Decide", key="ai_agent_button"):
    if st.session_state.prompt:
        with st.spinner("AI Agent is thinking..."):
            result = ai_agent(
                st.session_state.prompt, sd_model, music_model, processor, device, img_size
            )
            for msg in result["messages"]:
                st.markdown(f"> {msg}")
            st.session_state.generated_image = result.get("image")
            st.session_state.generated_music = result.get("music")
    else:
        st.warning("Please enter a prompt!")

# --- OUTPUT DISPLAY ---
if st.session_state.generated_image:
    st.image(st.session_state.generated_image, caption="Generated Image")

if st.session_state.generated_music:
    st.audio(st.session_state.generated_music, format="audio/wav")
