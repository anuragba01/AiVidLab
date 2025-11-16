# Bytez (Stable Diffusion XL) Fallback Integration

## Overview
The `ImageGenerator` now supports automatic fallback to Bytez (Stability AI's Stable Diffusion XL) when Gemini image generation fails.

## Setup Steps

### 1. Install Bytez
```bash
pip install bytez
```

### 2. Get Your Bytez API Key
1. Sign up at https://bytez.ai/ (or your Bytez provider)
2. Create an API key
3. Copy the key

### 3. Add to Your Environment
Create or update your `.env` file:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
BYTEZ_API_KEY=your_bytez_api_key_here
```

### 4. How It Works

**Primary Flow:**
1. Pipeline attempts image generation using **Gemini** (as configured in `config.json`)
2. If Gemini fails (404, timeout, rate limit, etc.), automatically falls back to **Bytez**
3. Bytez uses **Stable Diffusion XL** to generate high-quality images
4. Both Gemini and Bytez outputs are converted to PNG for consistency

**Fallback Triggers:**
- Gemini model not found (404 NOT_FOUND)
- Gemini API error or timeout
- Gemini returns empty response
- Any other Gemini exception

**If Bytez Also Fails:**
- Error is logged and the prompt is skipped
- Video generation continues with placeholder or falls back to royalty-free images

## Configuration

### Minimal Setup (Optional Bytez)
If `BYTEZ_API_KEY` is not set, Bytez fallback is disabled but the pipeline continues normally.

### Override Per Call
In code, you can pass a custom Bytez key:
```python
from processors.image_generator import ImageGenerator

gen = ImageGenerator(
    api_key=gemini_key,
    model_name="gemini-2.0-flash-preview-image-generation",
    bytez_api_key="your_custom_bytez_key"
)
```

## Verification

To test the setup:

```bash
# Test Gemini (will attempt Bytez fallback if Gemini fails)
python src/processors/image_generator.py
```

Expected output:
```
Initializing ImageGenerator with model: gemini-2.0-flash-preview-image-generation...
Bytez (Stable Diffusion XL) initialized as fallback.
ImageGenerator initialized successfully.
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'bytez'"
```bash
pip install bytez
```

### "Bytez model not initialized"
Check that `BYTEZ_API_KEY` is set in your environment:
```bash
echo $BYTEZ_API_KEY
```

### Both Gemini and Bytez Failing
- Verify API keys are valid and have remaining quota
- Check internet connectivity
- Review API documentation for rate limits
- Consider using royalty-free image fallback instead

## Performance Notes

- **Gemini**: Usually faster (~2-5s)
- **Bytez (Stable Diffusion XL)**: Slightly slower but high quality (~10-15s)
- Both outputs are PNG-converted for consistency

## Files Modified

- `requirements.txt`: Added `bytez`
- `src/processors/image_generator.py`: Added fallback logic
- `src/orchestrator.py`: Passes `BYTEZ_API_KEY` to ImageGenerator

## References

- Gemini API: https://ai.google.dev/
- Bytez: https://bytez.ai/
- Stable Diffusion XL: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
