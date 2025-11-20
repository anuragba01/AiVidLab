import logging
import os
from dotenv import load_dotenv

from src.processors.script_generator import ScriptGenerator

# --- Independent Test Block ---
# This code only runs when you execute this file directly (e.g., `python src/processors/script_generator.py`)
# Set up a logger for this module. This is better than print() for production.
logger = logging.getLogger(__name__)

if __name__ == '__main__':
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
    if script:
        logger.info("---  Test SUCCESS ---")
        print("\n--- Generated Script ---")
        print(script)
        print("------------------------")
    else:
        logger.error("--- Test FAILED: The process returned None. ---")