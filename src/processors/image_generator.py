"""
Image Generator Processor Module

This file contains the ImageGenerator class, a dedicated tool for creating
an image from a text prompt using Gemini vision or image generation model,
with fallback to Stability AI's Stable Diffusion via Bytez.

Responsibilities:
- Initialize the Gemini client with an image generation model.
- Take a text prompt as input.
- Call the Gemini API to generate an image (primary).
- Fall back to Bytez (Stability AI Stable Diffusion) on Gemini failure.
- Handle API responses and extract image data.
- Return the image data as bytes.
"""
import os
import time
import traceback
from io import BytesIO

# Necessary import for the _convert_to_png function
from PIL import Image

# Use the specific imports from your example
from google import genai
from google.genai import types

try:
    from bytez import Bytez
    BYTEZ_AVAILABLE = True
except ImportError:
    BYTEZ_AVAILABLE = False

class ImageGenerator:
    """
    Generates an image from a text prompt using Gemini API with Bytez fallback.
    """
    def __init__(self, api_key: str, model_name: str, bytez_api_key: str = None):
        """
        Initializes the ImageGenerator with a specific Gemini model and optional Bytez fallback.

        Args:
            api_key (str): The Google AI Studio API key.
            model_name (str): The specific Gemini model to use for image generation.
            bytez_api_key (str, optional): The Bytez/Stability AI API key for fallback. If not provided, will try to read from BYTEZ_API_KEY env var.
        """
        if not api_key:
            raise ValueError("API key must be provided for ImageGenerator.")
        if not model_name:
            raise ValueError("A model name must be provided for ImageGenerator.")

        print(f"Initializing ImageGenerator with model: {model_name}...")
        
        # Configure the Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # Initialize Bytez fallback if available
        self.bytez_sdk = None
        self.bytez_model = None
        self.gemini_has_failed = False
        if BYTEZ_AVAILABLE:
            bytez_key = bytez_api_key or os.getenv("BYTEZ_API_KEY")
            if bytez_key:
                try:
                    self.bytez_sdk = Bytez(bytez_key)
                    self.bytez_model = self.bytez_sdk.model("stabilityai/stable-diffusion-xl-base-1.0")
                    print("Bytez (Stable Diffusion XL) initialized as fallback.")
                except Exception as e:
                    print(f"Warning: Could not initialize Bytez fallback: {e}")
        
        print("ImageGenerator initialized successfully.")

    def process(self, prompt: str, negative_prompt: str = "", api_timeout_s: int = 180) -> bytes:
        """
        Generates an image from a prompt and returns the image data as bytes.
        Tries Gemini first. If it fails, it uses Bytez for all subsequent calls.

        Args:
            prompt (str): The descriptive text prompt for the image.
            negative_prompt (str, optional): A string of terms to avoid.
            api_timeout_s (int): The timeout in seconds for the API request.

        Returns:
            bytes: The raw PNG image data. Returns None on failure.
        """
        if not prompt or not prompt.strip():
            print("Warning (ImageGenerator): Input prompt is empty. Skipping generation.")
            return None

        # Append negative prompts directly to the main prompt
        full_prompt = prompt
        if negative_prompt and negative_prompt.strip():
            full_prompt = f"{prompt}. Avoid: {negative_prompt.strip()}"

        # Primary generator is Gemini, unless it has failed before.
        if not self.gemini_has_failed:
            print(f"Generating image for prompt: '{full_prompt[:100]}...'")
            try:
                image_bytes = self._generate_with_gemini(full_prompt)
                if image_bytes:
                    return image_bytes
                # Fallthrough to fallback if no image bytes returned
            except Exception as e:
                print(f"Gemini generation failed: {e}. Attempting fallback to Bytez (Stable Diffusion)...")
            
            # If we reach here, Gemini has failed one way or another.
            self.gemini_has_failed = True
        
        # Fallback generator is Bytez.
        # This is used if Gemini is not tried, or if it failed.
        if self.bytez_model:
            print("Using Bytez fallback generator.")
            try:
                image_bytes = self._generate_with_bytez(prompt)
                if image_bytes:
                    return image_bytes
            except Exception as e:
                print(f"Bytez fallback also failed: {e}")
        
        print("ERROR (ImageGenerator): Image generation failed with all available generators.")
        return None

    def _generate_with_gemini(self, full_prompt: str) -> bytes:
        """
        Generate image using Gemini API.
        
        Args:
            full_prompt (str): The full prompt including negative terms.
            
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
            print(f"Gemini Image Gen API call completed in {duration:.2f} seconds.")

            # Extract the image from the response
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and 'image' in part.inline_data.mime_type:
                        image_bytes = part.inline_data.data
                        print(f"Successfully received {len(image_bytes)} bytes of image data ({part.inline_data.mime_type}).")
                        return self._convert_to_png(image_bytes)
            
            # Handle cases where no image was found
            block_reason = "No image data in response."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                block_reason = response.prompt_feedback.block_reason
            raise ValueError(f"API response did not contain valid image data. Block reason: {block_reason}")

        except Exception as e:
            print(f"ERROR (ImageGenerator/Gemini): {e}")
            traceback.print_exc()
            raise

    def _generate_with_bytez(self, prompt: str) -> bytes:
        """
        Generate image using Bytez (Stability AI Stable Diffusion).
        
        Args:
            prompt (str): The text prompt for image generation.
            
        Returns:
            bytes: PNG image data or None on failure.
        """
        if not self.bytez_model:
            raise ValueError("Bytez model not initialized.")
        
        try:
            print(f"Generating image using Bytez (Stable Diffusion XL)...")
            start_time = time.time()
            
            output, error = self.bytez_model.run(prompt)
            
            duration = time.time() - start_time
            print(f"Bytez Image Gen API call completed in {duration:.2f} seconds.")
            
            if error:
                raise ValueError(f"Bytez API returned error: {error}")
            
            if not output:
                raise ValueError("Bytez API returned empty output.")
            
            # output should be image bytes; convert to PNG if needed
            image_bytes = self._convert_to_png(output)
            print(f"Successfully received {len(image_bytes)} bytes of image data from Bytez.")
            return image_bytes
            
        except Exception as e:
            print(f"ERROR (ImageGenerator/Bytez): {e}")
            traceback.print_exc()
            raise

    def _convert_to_png(self, image_bytes: bytes) -> bytes:
        """
        Ensures the final image data is in PNG format for consistency.
        Handles various image formats (PNG, JPEG, WebP, etc.).
        """
        if not image_bytes:
            return None
        
        try:
            img = Image.open(BytesIO(image_bytes))
            
            if img.format and img.format.upper() == 'PNG':
                return image_bytes
            
            # Convert any format to PNG
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            print(f"Warning (ImageGenerator): Could not convert image to PNG. Attempting to return original bytes. Error: {e}")
            # If conversion fails, try to return original bytes as-is
            return image_bytes if image_bytes else None

# --- Independent Test Block ---
if __name__ == '__main__':
    print("\n--- Running Independent Test for ImageGenerator ---")
    
    test_api_key = os.getenv("GEMINI_API_KEY")
    test_bytez_key = os.getenv("BYTEZ_API_KEY")

    if not test_api_key:
        print("\nWARNING: Please set the GEMINI_API_KEY environment variable for testing.")
        print("Skipping live API test.")
    else:
        # Use the model name from your example
        TEST_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"
        
        TEST_PROMPT = "An ancient scroll unfurling on a dark wooden table, a single, elegant ink brush stroke in the sumi-e style is visible, soft, cinematic lighting from a single candle."
        TEST_NEGATIVE_PROMPT = "photo, realistic, 3d render, text, signature"

        try:
            # 1. Instantiate the tool (with optional Bytez key for fallback)
            image_tool = ImageGenerator(
                api_key=test_api_key,
                model_name=TEST_MODEL_NAME,
                bytez_api_key=test_bytez_key
            )

            # 2. Use the tool to process the data
            generated_image_bytes = image_tool.process(
                prompt=TEST_PROMPT,
                negative_prompt=TEST_NEGATIVE_PROMPT
            )

            # 3. Save the output to verify
            if generated_image_bytes:
                output_filename = "test_image_output.png"
                with open(output_filename, "wb") as f:
                    f.write(generated_image_bytes)
                print(f"\nSUCCESS: Test image file saved to '{output_filename}'")
                print("You can now open this file to verify the output.")
            else:
                print("\nFAILURE: The processor returned None, indicating an error during generation.")

        except Exception as e:
            print(f"\nFAILURE: The test failed with an error: {e}")