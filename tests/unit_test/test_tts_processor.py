import logging
import pytest
from src.processors.tts_processor import TTSProcessor
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

def test_tts_processing():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    load_dotenv()

    # Configuration from the user's second code snippet
    TEST_MODEL = "gemini-2.5-flash-preview-tts"
    TEST_VOICE = "Kore"
    TEST_TEXT = "Say cheerfully: Have a wonderful day!"
    OUTPUT_FILENAME = "out.wav"

    try:
        tts = TTSProcessor(model_name=TEST_MODEL)
        audio_bytes = tts.process(TEST_TEXT, TEST_VOICE)

        assert audio_bytes is not None, "No audio data returned."
        
        tts.wave_file(OUTPUT_FILENAME, audio_bytes)
        logger.info(f"Audio file saved as {OUTPUT_FILENAME} ({len(audio_bytes)} bytes).")

    except (ValueError, EnvironmentError, RuntimeError) as e:
        logger.error(f"An error occurred: {e}")
        pytest.fail(f"TTS processing failed with exception: {e}")

# --- Test Block ---
if __name__ == "__main__":
    test_tts_processing()