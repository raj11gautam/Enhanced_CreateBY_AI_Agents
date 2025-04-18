from PIL import Image

def generate_image(prompt, sd_model, size=(512, 512)):
    image = sd_model(prompt, height=384, width=384).images[0]
    resized = image.resize(size)
    return resized
