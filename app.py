import streamlit as st
from diffusers import StableDiffusionPipeline
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
from utils.image_generator import generate_image
from utils.music_generator import generate_music  
from utils.ui_config import apply_custom_css
from refiner_agent import refine_prompt_with_feedback  # Importing refine function
from ai_agent import ai_agent  # Re-adding AI Agent import

# ---- SESSION STATE INIT ----
for key in ["show_sidebar", "prompt", "prompt_history", "generated_image", "generated_music", "refined_prompt", "refined_prompt_updated"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "show_sidebar" else [] if key == "prompt_history" else "" if key == "prompt" else None

# ---- SIDEBAR TOGGLE ----
if st.button("\u2630 Menu"):
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

# ---- MAIN CONTENT ----
st.title("🎨 AI Image & 🎵 Music Generator")

# Prompt input box where the refined prompt will be displayed after refinement
if "refined_prompt" in st.session_state and st.session_state.refined_prompt:
    prompt = st.text_area("Enter your prompt:", st.session_state.refined_prompt, height=150)
else:
    prompt = st.text_area("Enter your prompt:", st.session_state.prompt, height=150)

description_type = st.selectbox("Choose description length:", ["short", "medium", "long"])

# Collect feedback for refinement
feedback = {}

color = st.radio("Do you want to specify the color?", ["No", "Yes"])
if color == "Yes":
    feedback["color"] = st.text_input("Enter the desired color:")

size = st.radio("Do you want to specify the size?", ["No", "Yes"])
if size == "Yes":
    feedback["size"] = st.text_input("Enter the desired size:")

shape = st.radio("Do you want to specify the shape?", ["No", "Yes"])
if shape == "Yes":
    feedback["shape"] = st.text_input("Enter the desired shape:")

texture = st.radio("Do you want to specify the texture?", ["No", "Yes"])
if texture == "Yes":
    feedback["texture"] = st.text_input("Enter the desired texture:")

material = st.radio("Do you want to specify the material?", ["No", "Yes"])
if material == "Yes":
    feedback["material"] = st.text_input("Enter the desired material:")

# Refine the prompt based on feedback, only if it hasn't been updated already
if st.button("Refine Prompt"):
    if prompt and not st.session_state.refined_prompt_updated:  # Check if prompt hasn't been updated yet
        refined_prompt = refine_prompt_with_feedback(prompt, feedback, description_type)
        st.session_state.refined_prompt = refined_prompt  # Store the refined prompt
        st.session_state.refined_prompt_updated = True  # Mark as updated
    elif st.session_state.refined_prompt_updated:
        st.warning("Prompt has already been refined!")
    else:
        st.warning("Please enter a prompt first!")

# ---- GENERATE BUTTONS ----
if st.button("Generate Image"):
    if prompt:
        st.session_state.generated_image = generate_image(st.session_state.refined_prompt, sd_model)
        st.success("Image generated successfully!")
    else:
        st.warning("Please enter a prompt first!")

if st.button("Generate Music"):
    if prompt:
        st.session_state.generated_music = generate_music(st.session_state.refined_prompt, music_model, processor)
        st.success("Music generated successfully!")
    else:
        st.warning("Please enter a prompt first!")

if st.button("🎯 Let AI Agent Decide"):
    if prompt:
        with st.spinner("AI Agent is thinking..."):
            result = ai_agent(st.session_state.refined_prompt, sd_model, music_model, processor, device)
            for msg in result["messages"]:
                st.markdown(f"> {msg}")
            st.session_state.generated_image = result.get("image")
            st.session_state.generated_music = result.get("music")
    else:
        st.warning("Please enter a prompt first!")

# ---- OUTPUTS ----
if st.session_state.generated_image:
    st.image(st.session_state.generated_image, caption="Generated Image")

if st.session_state.generated_music:
    st.audio(st.session_state.generated_music, format="audio/wav")
