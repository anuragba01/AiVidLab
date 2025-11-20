import logging
import pytest
from dotenv import load_dotenv

from src.processors.script_generator import ScriptGenerator

# --- Independent Test Block ---
# This code only runs when you execute this file directly (e.g., `python src/processors/script_generator.py`)
# Set up a logger for this module. This is better than print() for production.
logger = logging.getLogger(__name__)


def test_script_generation():
    # --- Setup for Testing ---
    # Configure basic logging to see the output from the class
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    load_dotenv()

    # --- Test Execution ---
    logger.info("--- Running Independent Test for ScriptGenerator ---")
    
    # Instantiate the tool with the model you want to test
    tool = ScriptGenerator("gemini-2.0-flash-lite")

    # Call the processor with test data
    script = tool.process(
        topics=["The philosophy of Stoicism", "Practical applications in modern life"],
        keywords=["Marcus Aurelius", "resilience", "virtue"],
        tone="calm, inspirational",
        target_word_count=50
    )

    # --- Verification ---
    assert script is not None, "The process returned None."
    assert len(script) > 0, "The generated script is empty."
    
    logger.info("---  Test SUCCESS ---")
    print("\n--- Generated Script ---")
    print(script)
    print("------------------------")

if __name__ == '__main__':
    test_script_generation()