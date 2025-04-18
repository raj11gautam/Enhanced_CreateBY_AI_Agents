from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Function to generate prompt based on user input
def refine_prompt(original_prompt, keywords, max_length=50):
    # Ensure keywords is a list of strings
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    else:
        raise ValueError("Keywords should be a string of comma-separated values.")
    
    # Check if the original prompt is valid
    if not original_prompt.strip():
        raise ValueError("Original prompt is empty! Please provide a valid prompt.")
    
    # Build controlled refining instruction
    refining_instruction = (
        f"Refine this prompt: '{original_prompt}' by adding the keywords: {', '.join(keywords)}."
    )

    print("Refining Instruction: ", refining_instruction)  # Debugging line

    # Tokenize input
    inputs = tokenizer(refining_instruction, return_tensors="pt")

    # Debugging line to check the tokenized inputs
    print("Tokenized Inputs: ", inputs)

    # Generate based on the refined prompt
    try:
        outputs = model.generate(
            **inputs,
            max_length=int(max_length),
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.85,
            no_repeat_ngram_size=2,
            do_sample=False,
        )
    except Exception as e:
        print(f"Error during prompt generation: {e}")
        raise

    refined = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Debugging line to check the refined result
    print("Refined Output: ", refined)
    
    return refined
