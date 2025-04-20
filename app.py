import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from utils.image_generator import generate_image
from utils.music_generator import generate_music
from utils.ui_config import apply_custom_css
from ai_agent import ai_agent
from refiner_agent import AdvancedAIAgent
import io

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI Generator", layout="wide")
apply_custom_css()

# ---- SESSION STATE INIT ----
for key in ["show_sidebar", "prompt", "prompt_history", "generated_image", "generated_music", "refined_prompt", 
            "show_keyword_input", "refine_keywords", "generation_type", "prompt_style", "enhancement_level"]:
    if key in ["show_sidebar", "show_keyword_input"]:
        if key not in st.session_state:
            st.session_state[key] = False
    elif key == "prompt_history":
        if key not in st.session_state:
            st.session_state[key] = []
    elif key == "generation_type":
        if key not in st.session_state:
            st.session_state[key] = "image"
    elif key == "prompt_style":
        if key not in st.session_state:
            st.session_state[key] = "realistic"
    elif key == "enhancement_level":
        if key not in st.session_state:
            st.session_state[key] = 1
    else:
        if key not in st.session_state:
            st.session_state[key] = ""

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
    
if "image_size" not in st.session_state:
    st.session_state.image_size = 256  # Default image size
    
if "show_image_slider" not in st.session_state:
    st.session_state.show_image_slider = False

# Toggle image slider visibility
def toggle_image_slider():
    st.session_state.show_image_slider = not st.session_state.show_image_slider

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
user_input = st.text_area("Enter your prompt:", st.session_state.get("prompt", ""), height=150, key="prompt_input_box")
st.session_state.prompt = user_input

# Place all buttons horizontally side by side
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🔁 Refine Prompt", key="refine_button_main", use_container_width=True):
        st.session_state.show_keyword_input = not st.session_state.show_keyword_input

with col2:
    if st.button("🎨 Generate Image", key="generate_image_button", use_container_width=True, on_click=toggle_image_slider):
        if not st.session_state.prompt:
            st.warning("Please enter a prompt!")

with col3:
    if st.button("🎵 Generate Music", key="generate_music_button", use_container_width=True):
        if st.session_state.prompt:
            with st.spinner("Generating music..."):
                st.session_state.generated_music = generate_music(st.session_state.prompt, music_model, processor)
            st.success("Music generated successfully!")
        else:
            st.warning("Please enter a prompt!")

with col4:
    if st.button("🤖 Let AI Agent Decide", key="ai_agent_button", use_container_width=True):
        if st.session_state.prompt:
            with st.spinner("AI Agent is thinking..."):
                img_size = (st.session_state.image_size, st.session_state.image_size)
                result = ai_agent(
                    st.session_state.prompt, sd_model, music_model, processor, device, img_size
                )
                for msg in result["messages"]:
                    st.markdown(f"> {msg}")
                st.session_state.generated_image = result.get("image")
                st.session_state.generated_music = result.get("music")
        else:
            st.warning("Please enter a prompt!")

# Show image size slider only when show_image_slider is True
if st.session_state.show_image_slider:
    st.subheader("Image Size Settings")
    
    # Display current size value before slider
    current_size = st.session_state.image_size
    st.write(f"**Current image size:** {current_size} x {current_size} pixels")
    
    # Update the image size with slider
    st.session_state.image_size = st.slider(
        "Adjust Image Size:", 
        min_value=256, 
        max_value=512, 
        value=current_size,
        step=64,
        key="image_size_slider"
    )
    
    # Show the selected size after adjustment
    st.write(f"**Selected image size:** {st.session_state.image_size} x {st.session_state.image_size} pixels")
    
    img_size = (st.session_state.image_size, st.session_state.image_size)
    
    if st.button("✅ Confirm and Generate Image", key="confirm_generate_image"):
        if st.session_state.prompt:
            with st.spinner("Generating image..."):
                image = generate_image(st.session_state.prompt, sd_model, img_size)
                st.session_state.generated_image = image
                st.session_state.generated_images.append(image)  # Add to history
            st.success("Image generated successfully!")
            st.session_state.show_image_slider = False  # Hide slider after generating
        else:
            st.warning("Please enter a prompt!")

# Advanced Prompt Refinement Section
if st.session_state.show_keyword_input:
    with st.expander("Advanced Prompt Refinement", expanded=True):
        st.markdown("### AI Prompt Refiner")
        st.session_state.refine_keywords = st.text_input("Enter keywords (comma-separated):", 
                                                        key="keyword_input_box")

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.generation_type = st.selectbox(
                "Generation Type:", 
                ["image", "story", "music"],
                key="generation_type_select"
            )

        with col2:
            style_options = {
                "image": ["realistic", "anime", "abstract"],
                "story": ["scifi", "fantasy"],
                "music": ["classical", "electronic", "ambient"]
            }
            selected_styles = style_options.get(st.session_state.generation_type, ["realistic"])

            st.session_state.prompt_style = st.selectbox(
                "Prompt Style:", 
                selected_styles,
                key="prompt_style_select"
            )

        # Replace selectbox with a slider for description length
        st.session_state.description_type = st.slider(
            "Description Length:", 
            min_value=1, 
            max_value=3, 
            value=1, 
            step=1,
            key="description_slider"
        )

        # Mapping slider value to description length
        description_map = {
            1: "short",
            2: "medium",
            3: "long"
        }
        description_length = description_map[st.session_state.description_type]

        # Displaying the selected description length
        st.write(f"Selected Description Length: {description_length.capitalize()}")
        
        # Initialize feedback state if not already present
        if "feedback_state" not in st.session_state:
            st.session_state.feedback_state = "initial"  # States: initial, prompt_generated, refining
        
        if "refinement_count" not in st.session_state:
            st.session_state.refinement_count = 0
            
        # Generate button - only show if in initial state
        if st.session_state.feedback_state == "initial":
            if st.button("✅ Generate Refined Prompt", key="generate_refined_prompt_button"):
                with st.spinner("Refining prompt..."):
                    agent = AdvancedAIAgent()
                    refined = agent.generate_prompt(
                        st.session_state.refine_keywords,
                        st.session_state.generation_type,
                        st.session_state.prompt_style,
                        description_length
                    )
                    st.session_state.prompt = refined
                    st.session_state.refined_prompt = refined
                    st.session_state.show_keyword_input = True
                    st.session_state.feedback_state = "prompt_generated"
                    st.session_state.refinement_count = 1
                
                st.success("Prompt refined using your keywords!")
                st.rerun()
        
        # Auto-refine when feedback is negative
        elif st.session_state.feedback_state == "feedback_no":
            with st.spinner(f"Refining prompt (attempt #{st.session_state.refinement_count})..."):
                agent = AdvancedAIAgent()
                refined = agent.generate_prompt(
                    st.session_state.refine_keywords,
                    st.session_state.generation_type,
                    st.session_state.prompt_style,
                    description_length,
                    st.session_state.refined_prompt,  # Pass previous prompt
                    st.session_state.refinement_count  # Pass refinement count
                )
                st.session_state.prompt = refined
                st.session_state.refined_prompt = refined
                st.session_state.feedback_state = "prompt_generated"
            
            st.success(f"Prompt refined (attempt #{st.session_state.refinement_count})!")
            st.rerun()
        
        # Show the refined prompt and feedback buttons if in prompt_generated state
        if st.session_state.feedback_state == "prompt_generated" and st.session_state.refined_prompt:
            st.write("**Refined Prompt:**")
            st.info(st.session_state.refined_prompt)
            
            # Feedback buttons in two columns
            feedback_col1, feedback_col2 = st.columns(2)
            with feedback_col1:
                if st.button("👍 Yes, I like it!", key="feedback_yes"):
                    st.session_state.feedback_state = "feedback_yes"
                    st.success("Great! Your refined prompt is ready to use.")
                    st.rerun()
                    
            with feedback_col2:
                if st.button("👎 No, refine again", key="feedback_no"):
                    st.session_state.feedback_state = "feedback_no"
                    st.session_state.refinement_count += 1
                    st.rerun()  # Immediately rerun to trigger auto-refinement
        
        # After positive feedback, just show the prompt
        elif st.session_state.feedback_state == "feedback_yes" and st.session_state.refined_prompt:
            st.write("**Your Approved Prompt:**")
            st.info(st.session_state.refined_prompt)
            
            # Option to start over
            if st.button("Start Over with New Keywords", key="start_over"):
                st.session_state.feedback_state = "initial"
                st.session_state.refinement_count = 0
                st.session_state.refined_prompt = ""
                st.rerun()

# Save prompt in history
if st.session_state.prompt and (
    len(st.session_state.prompt_history) == 0 or st.session_state.prompt_history[-1] != st.session_state.prompt
):
    st.session_state.prompt_history.append(st.session_state.prompt)

# --- OUTPUT DISPLAY ---
st.markdown("## Generated Output")
output_tabs = st.tabs(["Image", "Music"])

with output_tabs[0]:
    if st.session_state.generated_image:
        # You can adjust the width (e.g., 500px or a custom value)
        st.image(st.session_state.generated_image, caption="Generated Image", width=250)
    else:
        st.info("Generate an image to see the result here.")

with output_tabs[1]:
    if st.session_state.generated_music:
        st.audio(st.session_state.generated_music, format="audio/wav")
    else:
        st.info("Generate music to hear the result here.")

# --- ALL GENERATED IMAGES --- 
st.markdown("## 🖼️ All Generated Images")

if st.session_state.generated_images:
    images = list(reversed(st.session_state.generated_images))
    num_per_row = 3
    for i in range(0, len(images), num_per_row):
        cols = st.columns(num_per_row)
        for j, img in enumerate(images[i:i+num_per_row]):
            with cols[j]:
                # Display thumbnail
                img_size_text = f"{img.width} x {img.height}"
                st.image(img.resize((150, 150)), caption=f"Image #{len(images) - (i + j)} ({img_size_text})")

                # Download button
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                byte_img = buf.getvalue()
                st.download_button(
                    label="⬇️ Download",
                    data=byte_img,
                    file_name=f"generated_image_{len(images) - (i + j)}.png",
                    mime="image/png",
                    key=f"download_{i}_{j}"
                )
else:
    st.info("Generate images and they'll appear here.")
