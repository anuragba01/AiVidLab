import logging
from src.processors.tts_processor import TTSProcessor
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

# --- Test Block ---
if __name__ == "__main__":
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

        if audio_bytes:
            tts.wave_file(OUTPUT_FILENAME, audio_bytes)
            logger.info(f"Audio file saved as {OUTPUT_FILENAME} ({len(audio_bytes)} bytes).")
        else:
            logger.warning("No audio data returned.")

    except (ValueError, EnvironmentError, RuntimeError) as e:
        logger.error(f"An error occurred: {e}")