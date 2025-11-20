"""
Prompt Processor Module

Generates a concise, visually descriptive text prompt for an image generation model.
Uses a Gemini LLM to interpret input text, creative brief, and overall context.
"""

import os
import logging
from google import genai

logger = logging.getLogger(__name__)

class PromptProcessor:
    """Generates image prompts by interpreting text within a creative and thematic context."""

    def __init__(self, model_name: str):
        if not model_name:
            raise ValueError("A model name must be provided for PromptProcessor.")

        if not os.getenv("GEMINI_API_KEY"):
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

        self.model_name = model_name
        logger.info(f"PromptProcessor initialized with model: {model_name}")

    def process(self, text_chunk: str, creative_brief: str, global_summary: str) -> str:
        """Generates a concise, descriptive prompt for text-to-image AI."""
        if not text_chunk or not text_chunk.strip():
            logger.warning("Empty text_chunk provided; returning fallback prompt.")
            return "A simple neutral background scene."

        instructional_prompt = (
            "You are a visual concept artist. Generate one concise, visually descriptive prompt "
            "for a text-to-image AI, focusing on what can be seen.\n\n"
            f"Overall Theme: {global_summary}\n"
            f"Creative Brief: {creative_brief}\n"
            f"Specific Text: {text_chunk}\n\n"
            "Describe concrete visuals (lighting, subjects, environment, mood). "
            "Avoid abstract concepts or commentary. "
            "Output only a single descriptive paragraph (30–70 words)."
        )

        try:
            client = genai.Client()
            

            response = client.models.generate_content(
                model=self.model_name,
                contents=instructional_prompt,
            )

            try:
                generated_prompt = response.candidates[0].content.parts[0].text.strip()
            except (AttributeError, IndexError):
                generated_prompt = None

            if generated_prompt:
                logger.info("Prompt generated successfully.")
                return generated_prompt

            logger.warning("LLM returned empty response; using fallback prompt.")
            return f"A cinematic, visually rich depiction of: {text_chunk}"

        except Exception:
            logger.exception("PromptProcessor failed during API call.")
            return f"A visually detailed interpretation of: {text_chunk}"

    def generate_image_prompt(self, text_chunk: str, creative_brief: str, global_summary: str) -> str:
        """Explicit API for generating prompts intended for the image generator.

        Kept separate for clarity in orchestrator: this function returns the prompt
        string that should be passed directly to `ImageGenerator.process`.
        """
        # Reuse existing `process` implementation to avoid duplicating LLM logic.
        return self.process(text_chunk=text_chunk, creative_brief=creative_brief, global_summary=global_summary)


