# ai_agent.py
from utils.image_generator import generate_image
from utils.music_generator import generate_music

def ai_agent(prompt, sd_model, music_model, processor, device, img_size):
    lower_prompt = prompt.lower()
    response = {
        "messages": [],
        "image": None,
        "music": None
    }

    # Detect what to generate
    wants_image = any(word in lower_prompt for word in ["draw", "image", "picture", "art", "photo", "scene"])
    wants_music = any(word in lower_prompt for word in ["music", "melody", "song", "tune", "sound"])

    # Default to both if unclear
    if not wants_image and not wants_music:
        wants_image = wants_music = True
        response["messages"].append("🤖 I wasn't sure what you wanted, so I generated both image and music for you!")

    if wants_image:
        response["messages"].append("🎨 Generating a beautiful image for your prompt...")
        image = generate_image(prompt, sd_model, img_size)
        response["image"] = image
        response["messages"].append("✅ Image generated!")

    if wants_music:
        response["messages"].append("🎵 Creating a musical tune based on your input...")
        music = generate_music(prompt, music_model, processor)
        response["music"] = music
        response["messages"].append("✅ Music generated!")

    return response
