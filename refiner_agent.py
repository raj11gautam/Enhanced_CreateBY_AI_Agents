from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Function to refine prompt based on user feedback
def refine_prompt_with_feedback(prompt, feedback, description_type, max_length=50):
    refined_prompt = f"A {prompt}, showing key visual features like color and size."

    # Incorporating user feedback into the prompt
    if "color" in feedback:
        refined_prompt += f" The color of the {prompt} should be {feedback['color']}."
    if "size" in feedback:
        refined_prompt += f" The size of the {prompt} is {feedback['size']}."
    if "shape" in feedback:
        refined_prompt += f" The shape of the {prompt} is {feedback['shape']}."
    if "texture" in feedback:
        refined_prompt += f" The texture of the {prompt} is {feedback['texture']}."
    if "material" in feedback:
        refined_prompt += f" The material of the {prompt} is {feedback['material']}."

    # Depending on description length, generate detailed or short prompt
    if description_type == "short":
        refined_prompt = refined_prompt[:max_length]
    elif description_type == "medium":
        refined_prompt = refined_prompt[:max_length * 2]
    else:
        refined_prompt = refined_prompt[:max_length * 3]

    inputs = tokenizer(refined_prompt, return_tensors="pt")

    # Generate based on the refined prompt
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.85,
        no_repeat_ngram_size=2,
        do_sample=False,
    )

    refined = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return refined
