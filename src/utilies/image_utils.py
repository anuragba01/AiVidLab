import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

def convert_to_png(image_bytes: bytes) -> bytes:
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
        logger.warning(f"Could not convert image to PNG. Attempting to return original bytes. Error: {e}")
        # If conversion fails, try to return original bytes as-is
        return image_bytes if image_bytes else None
