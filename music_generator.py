import numpy as np
from scipy.io.wavfile import write as write_wav
from io import BytesIO

def generate_music(prompt, model, processor):
    inputs = processor(text=[prompt], return_tensors="pt").to("cpu")
    audio_values = model.generate(**inputs, max_new_tokens=256)
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_tensor = audio_values[0].cpu()

    audio_np = audio_tensor.numpy()
    if len(audio_np.shape) == 2:
        audio_np = audio_np.T

    audio_np = audio_np / np.max(np.abs(audio_np))
    audio_np = (audio_np * 32767).astype(np.int16)

    buffer = BytesIO()
    write_wav(buffer, int(sampling_rate), audio_np)
    buffer.seek(0)

    return buffer.read()
