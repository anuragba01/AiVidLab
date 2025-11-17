import os
import logging
import requests
from io import BytesIO

try:
    from bytez import Bytez
    BYTEZ_AVAILABLE = True
except ImportError:
    BYTEZ_AVAILABLE = False

from .image_utils import convert_to_png

logger = logging.getLogger(__name__)

def generate_image_with_bytez(prompt: str, bytez_api_key: str, timeout: int = 180) -> bytes:
    """
    Generates an image using Bytez (Stability AI Stable Diffusion).
    
    Args:
        prompt (str): The text prompt for image generation.
        bytez_api_key (str): The Bytez API key.
        timeout (int): The timeout in seconds for the API request.
        
    Returns:
        bytes: PNG image data or None on failure.
    """
    if not BYTEZ_AVAILABLE:
        logger.error("Bytez SDK not installed. Cannot use fallback image generator.")
        return None

    if not bytez_api_key:
        logger.error("Bytez API key not provided. Cannot use fallback image generator.")
        return None

    try:
        bytez_sdk = Bytez(bytez_api_key)
        bytez_model = bytez_sdk.model("stabilityai/stable-diffusion-xl-base-1.0")
        logger.info("Bytez (Stable Diffusion XL) initialized for fallback generation.")
    except Exception as e:
        logger.error(f"Could not initialize Bytez fallback: {e}", exc_info=True)
        return None

    try:
        logger.info(f"Generating image using Bytez (Stable Diffusion XL) for prompt: '{prompt[:100]}...'")
        
        result = bytez_model.run(prompt,{"width": 1024, "height": 576})
        
        if hasattr(result, 'error') and result.error:
            raise ValueError(f"Bytez API returned error: {result.error}")
        
        if not hasattr(result, 'output') or not result.output:
            raise ValueError("Bytez API returned empty output.")
        
        # result.output is a URL, download it
        response = requests.get(result.output, timeout=timeout)
        response.raise_for_status() # Raise an exception for HTTP errors
        
        image_bytes = response.content
        png_image_bytes = convert_to_png(image_bytes)
        logger.info(f"Successfully received {len(png_image_bytes)} bytes of image data from Bytez.")
        return png_image_bytes
        
    except Exception as e:
        logger.error(f"ERROR (FallbackGenerator/Bytez): {e}", exc_info=True)
        return None
