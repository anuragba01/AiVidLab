"""
Image Generator Processor Module

This file contains the ImageGenerator class, a dedicated tool for creating
an image from a text prompt using a Gemini vision or image generation model.

Responsibilities:
- Initialize the Gemini client with an image generation model.
- Take a text prompt as input.
- Call the Gemini API to generate an image.
- Handle the API response, extracting the raw image data.
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

class ImageGenerator:
    """
    Generates an image from a text prompt using the Gemini API.
    """
    def __init__(self, api_key: str, model_name: str):
        """
        Initializes the ImageGenerator with a specific Gemini model.

        Args:
            api_key (str): The Google AI Studio API key.
            model_name (str): The specific Gemini model to use for image generation.
        """
        if not api_key:
            raise ValueError("API key must be provided for ImageGenerator.")
        if not model_name:
            raise ValueError("A model name must be provided for ImageGenerator.")

        print(f"Initializing ImageGenerator with model: {model_name}...")
        
        # Configure the client as per the provided logic
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        print("ImageGenerator initialized successfully.")

    def process(self, prompt: str, negative_prompt: str = "", api_timeout_s: int = 180) -> bytes:
        """
        Generates an image from a prompt and returns the image data as bytes.

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

        print(f"Generating image for prompt: '{full_prompt[:100]}...'")
        
        try:
            start_time = time.time()
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=(full_prompt,),
                config=types.GenerateContentConfig(
                    # --- THIS IS THE FIX ---
                    # The API requires specifying both TEXT and IMAGE for this model
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
            print(f"ERROR (ImageGenerator): An exception occurred during the API call: {e}")
            traceback.print_exc()
            return None

    def _convert_to_png(self, image_bytes: bytes) -> bytes:
        """
        Ensures the final image data is in PNG format for consistency.
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            
            if img.format == 'PNG':
                return image_bytes
            
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            print(f"Warning (ImageGenerator): Could not convert image to PNG. Returning original bytes. Error: {e}")
            return image_bytes

# --- Independent Test Block ---
if __name__ == '__main__':
    print("\n--- Running Independent Test for ImageGenerator ---")
    
    test_api_key = os.getenv("GEMINI_API_KEY")

    if not test_api_key:
        print("\nWARNING: Please set the GEMINI_API_KEY environment variable for testing.")
        print("Skipping live API test.")
    else:
        # Use the model name from your example
        TEST_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"
        
        TEST_PROMPT = "An ancient scroll unfurling on a dark wooden table, a single, elegant ink brush stroke in the sumi-e style is visible, soft, cinematic lighting from a single candle."
        TEST_NEGATIVE_PROMPT = "photo, realistic, 3d render, text, signature"

        try:
            # 1. Instantiate the tool
            image_tool = ImageGenerator(api_key=test_api_key, model_name=TEST_MODEL_NAME)

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