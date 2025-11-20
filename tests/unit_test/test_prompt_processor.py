import logging
from src.processors.prompt_processor import PromptProcessor
from dotenv import load_dotenv


# Optional standalone test block (kept minimal)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    load_dotenv()
    
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
