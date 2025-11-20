from pydub.generators import Sine
from src.processors.audio_analyzer import AudioAnalyzer

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("Running standalone test for AudioAnalyzer...")

    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded .env file.")
   

    # ---------------------------
    #  Minimal test audio sample
    # ---------------------------
    # Creates ~0.5 sec of silent audio as raw bytes

    logger.info("Generating test sine wave...")
    test_audio = Sine(440).to_audio_segment(duration=500)
    audio_bytes = test_audio.export(format="wav").read()

    # Analyzer
    analyzer = AudioAnalyzer(model_size="tiny", device="cpu")

    # Fake chunk config
    config = {
        "min_silence_len_ms": 300,
        "min_chunk_duration_s": 0.1,
        "max_chunk_duration_s": 1.0
    }

    try:
        result = analyzer.process(audio_bytes, config)

        logger.info("Test SUCCESS")
        print("\nWord Timestamps:", result["word_timestamps"])
        print("\nPacing Chunks:", result["pacing_chunks"])

    except Exception as e:
        logger.error(f"Test FAILED: {e}")
