import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import nltk
from nltk.corpus import wordnet
from typing import List, Dict, Optional
import time

nltk.download('wordnet', quiet=True)

class AdvancedAIAgent:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        # Prompt templates
        self.templates = {
            "image": {
                "realistic": "Craft a {length} photorealistic image description focusing on lighting, texture, and fine details using: {keywords}.",
                "anime": "Design a {length} anime-style scene with expressive characters and dynamic backgrounds based on: {keywords}.",
                "abstract": "Imagine a {length} surreal abstract composition reflecting these ideas: {keywords}. Include symbolism and vibrant contrasts."
            },
            "story": {
                "scifi": "Construct a {length} sci-fi narrative prompt with futuristic technology and complex worldbuilding around: {keywords}.",
                "fantasy": "Invent a {length} fantasy tale with magic, mythical creatures, and heroism using the following: {keywords}."
            },
            "music": {
                "classical": "Design a {length} classical music composition prompt that evokes emotion and structure, based on: {keywords}.",
                "electronic": "Generate a {length} electronic music theme focusing on beats, synth layers, and mood using: {keywords}.",
                "ambient": "Describe a {length} ambient soundscape built with slow textures and immersive tones derived from: {keywords}."
            }
        }

        # Refinement templates for when feedback is negative
        self.refinement_templates = {
            "image": "Improve this image prompt by adding more details, alternative perspective, and enhancing visual elements: {previous_prompt}. Use these keywords: {keywords}.",
            "story": "Create a more engaging storyline based on this concept by adding conflict, character depth, and world-building: {previous_prompt}. Using these keywords: {keywords}.",
            "music": "Enhance this musical concept with more emotional range, instrument variety, and better atmosphere: {previous_prompt}. Based on these keywords: {keywords}."
        }

    def expand_keywords(self, keywords: List[str], callback=None) -> List[str]:
        """Enhance keywords with relevant synonyms and related terms, filtered by context"""
        expanded = []
        total_words = len(keywords)
        
        for i, word in enumerate(keywords):
            expanded.append(word)
            
            # Report progress if callback is provided
            if callback:
                progress = (i + 1) / total_words
                callback(progress * 0.3)  # First 30% of the process

            # Expand only relevant synonyms
            for syn in wordnet.synsets(word)[:2]:  # Only 2 synonyms
                for lemma in syn.lemmas():
                    # Only add synonyms that are related to the word context
                    if lemma.name().lower() not in expanded and lemma.name().lower() != word.lower():
                        # Additional check to keep terms relevant
                        if "yellow" in lemma.name().lower() or "man" in lemma.name().lower() or "tall" in lemma.name().lower():
                            expanded.append(lemma.name().replace('_', ' '))

        return list(set(expanded))[:10]  # Max 10 unique keywords

    def generate_prompt(self, keywords: str, genre: str = "image", style: str = "realistic", 
                       length: str = "short", previous_prompt: str = None, refinement_count: int = 1, 
                       callback=None) -> str:
        """Generate high-quality prompt based on keywords"""
        start_time = time.time()
        
        # Initialize progress
        if callback:
            callback(0.0)
            
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        if not keywords_list:
            return "Please provide valid keywords."

        # Enhance keywords with progress reporting
        enhanced_keywords = self.expand_keywords(keywords_list, callback)

        # Template injection
        length_label = {
            "short": "1-line",
            "medium": "2 to 3 line",
            "long": "detailed"
        }.get(length.lower(), "short")

        # If this is a refinement (not the first generation)
        if previous_prompt and refinement_count > 1:
            # Use refinement template
            template = self.refinement_templates.get(genre, "Improve this prompt: {previous_prompt}. Using: {keywords}.")
            input_text = template.format(
                previous_prompt=previous_prompt,
                keywords=", ".join(enhanced_keywords)
            )
            
            # For refinements, add instructional context
            if refinement_count == 2:
                input_text = f"Try a completely different approach than the previous attempt: {input_text}"
            elif refinement_count >= 3:
                input_text = f"Make this dramatically different from previous attempts - change perspective, style, and focus: {input_text}"
        else:
            # For first-time generation, use standard template
            template = self.templates.get(genre, {}).get(style, "Describe something based on: {keywords}.")
            input_text = template.format(length=length_label, keywords=", ".join(enhanced_keywords))

        # Update progress to 40%
        if callback:
            callback(0.4)

        # Set token count based on length parameter
        max_tokens_map = {
            "short": 60,
            "medium": 150,
            "long": 300
        }
        max_tokens = max_tokens_map.get(length.lower(), 60)

        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Update progress to 50%
        if callback:
            callback(0.5)

        # Generate with varying temperature based on refinement count(Higher temperature for refinements to encourage diversity)
        temperature = min(0.7 + (refinement_count * 0.1), 0.95)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )
        
        # Final stage - decoding and cleaning
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        cleaned_prompt = self._clean_prompt(decoded)
        
        # Complete the progress
        if callback:
            callback(1.0)
            
        total_time = time.time() - start_time
        print(f"Prompt generation completed in {total_time:.2f} seconds")
        
        return cleaned_prompt

    def _clean_prompt(self, text: str) -> str:
        """Extract only the clean refined prompt after 'output' or 'answer', fallback if needed."""
        # Try to extract after 'output:' or 'answer:'
        match = re.search(r"(?:output|answer)\s*[:\-]*\s*(.+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().capitalize()

        # Fallback cleanup
        text = re.sub(r"(Generate a .*? prompt.*?:|Create a .*? idea.*?:|Input.*?:|##output.*?:?)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().capitalize()
