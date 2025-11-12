"""
Prompt Processor Module

Generates a concise, visually descriptive text prompt for an image generation model.
Uses a Gemini LLM to interpret input text, creative brief, and overall context.
"""

import os
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


# Predefined safety settings (initialized once)
SAFETY_SETTINGS = [
    {"category": c, "threshold": types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE}
    for c in list(types.HarmCategory)
    if c != types.HarmCategory.HARM_CATEGORY_UNSPECIFIED
]


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

        except Exception as e:
            logger.exception("PromptProcessor failed during API call.")
            return f"A visually detailed interpretation of: {text_chunk}"


# Optional standalone test block (kept minimal)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    TEST_MODEL = "gemini-2.0-flash-lite"
    TEST_TEXT = (
        "The true measure of a man is not how he behaves in comfort, "
        "but how he stands in times of challenge and controversy."
    )
    TEST_BRIEF = "Cinematic, moody lighting, photorealistic style, soft contrast, introspective tone."
    TEST_SUMMARY = "A short film on Stoicism and emotional resilience."

    processor = PromptProcessor(TEST_MODEL)
    result = processor.process(TEST_TEXT, TEST_BRIEF, TEST_SUMMARY)

    print("\nGenerated Prompt:\n", result)
