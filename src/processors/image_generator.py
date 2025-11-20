"""
Image Generator Processor Module

This file contains the ImageGenerator class, a dedicated tool for creating
an image from a text prompt using Gemini vision or image generation model,
with fallback to a separate Bytez generator utility.

Responsibilities:
- Initialize the Gemini client with an image generation model.
- Take a text prompt as input.
- Call the Gemini API to generate an image.
- If Gemini fails, delegate all subsequent generation to a fallback utility.
- Handle API responses and extract image data.
- Return the image data as bytes.
"""
import os
import time
import logging

# Necessary import for the _convert_to_png function

# Use the specific imports from your example
from google import genai
from google.genai import types

from src.utilities.fallback_image_generator import generate_image_with_bytez
from src.utilities.image_utils import convert_to_png

logger = logging.getLogger(__name__)

class ImageGenerator:
    """
    Generates an image from a text prompt using Gemini API, with the ability
    to fail over to a Bytez-based fallback generator.
    """
    def __init__(self, api_key: str, model_name: str, bytez_api_key: str = None):
        """
        Initializes the ImageGenerator with a specific Gemini model.

        Args:
            api_key (str): The Google AI Studio API key.
            model_name (str): The specific Gemini model to use for image generation.
            bytez_api_key (str, optional): The Bytez/Stability AI API key for fallback.
        """
        if not api_key:
            raise ValueError("API key must be provided for ImageGenerator.")
        if not model_name:
            raise ValueError("A model name must be provided for ImageGenerator.")

        logger.info(f"Initializing ImageGenerator with model: {model_name}...")
        
        # Configure the Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # Store Bytez key for fallback, but do not initialize here
        self.bytez_api_key = os.getenv("BYTEZ_API_KEY")
        self.gemini_has_failed = False
        
        logger.info("ImageGenerator initialized successfully.")

    def process(self, prompt: str, negative_prompt: str = "", api_timeout_s: int = 180) -> bytes:
        """
        Generates an image from a prompt and returns the image data as bytes.
        Tries Gemini first. If it fails, it uses the Bytez helper for all subsequent calls.

        Args:
            prompt (str): The descriptive text prompt for the image.
            negative_prompt (str, optional): A string of terms to avoid.
            api_timeout_s (int): The timeout in seconds for the API request.

        Returns:
            bytes: The raw PNG image data. Returns None on failure.
        """
        if not prompt or not prompt.strip():
            logger.warning("Input prompt is empty. Skipping generation.")
            return None

        # Append negative prompts directly to the main prompt
        full_prompt = prompt
        if negative_prompt and negative_prompt.strip():
            full_prompt = f"{prompt}. Avoid: {negative_prompt.strip()}"

        # Primary generator is Gemini, unless it has failed before.
        if not self.gemini_has_failed:
            logger.info(f"Generating image with Gemini for prompt: '{full_prompt[:100]}...'")
            try:
                # Pass timeout to Gemini call
                image_bytes = self._generate_with_gemini(full_prompt, timeout=api_timeout_s)
                if image_bytes:
                    return image_bytes
                # Fallthrough to fallback if no image bytes returned
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}. Switching to fallback generator.")
            
            # If we reach here, Gemini has failed one way or another.
            self.gemini_has_failed = True
        
        # Fallback to Bytez generator utility.
        logger.info("Using fallback image generator (Bytez).")
        return generate_image_with_bytez(prompt, self.bytez_api_key, timeout=api_timeout_s)

    def _generate_with_gemini(self, full_prompt: str, timeout: int) -> bytes:
        """
        Generate image using Gemini API.
        
        Args:
            full_prompt (str): The full prompt including negative terms.
            timeout (int): The timeout in seconds for the API request.
            
        Returns:
            bytes: PNG image data or None on failure.
        """
        try:
            start_time = time.time()
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=(full_prompt,),
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"] 
                ),
            )
            
            duration = time.time() - start_time
            logger.info(f"Gemini Image Gen API call completed in {duration:.2f} seconds.")

            # Extract the image from the response
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and 'image' in part.inline_data.mime_type:
                        image_bytes = part.inline_data.data
                        logger.info(f"Successfully received {len(image_bytes)} bytes of image data ({part.inline_data.mime_type}).")
                        return convert_to_png(image_bytes)
            
            # Handle cases where no image was found
            block_reason = "No image data in response."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                block_reason = response.prompt_feedback.block_reason
            raise ValueError(f"API response did not contain valid image data. Block reason: {block_reason}")

        except Exception as e:
            logger.error(f"ERROR (ImageGenerator/Gemini): {e}", exc_info=True)
            raise

