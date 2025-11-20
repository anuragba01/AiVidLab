import logging
import pytest
from src.processors.prompt_processor import PromptProcessor
from dotenv import load_dotenv


def test_prompt_processing():
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

    assert result is not None, "Process returned None"
    assert isinstance(result, str), f"Expected a string, but got {type(result)}"
    print("\nGenerated Prompt:\n", result)

if __name__ == "__main__":
    test_prompt_processing()
