# Save this as src/processors/script_generator.py

import os
import sys
import time
import logging
import traceback
from typing import List
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    # This provides a clearer error message if the library isn't installed.
    raise ImportError(
        "The 'google-generativeai' library is required. "
        "Please install it with 'pip install google-generativeai'"
    )

# Set up a logger for this module. This is better than print() for production.
logger = logging.getLogger(__name__)

class ScriptGenerator:
    """
    Generates a video script using a Gemini LLM based on user-provided topics.
    """
    def __init__(self, model_name: str):
        """
        Initializes the ScriptGenerator.

        Args:
            model_name (str): The specific Gemini LLM to use (e.g., "gemini-pro").
            max_retries (int): The maximum number of times to retry the API call on failure.
        """
        if not model_name:
            raise ValueError("A model name must be provided for ScriptGenerator.")
            
        self.model_name = model_name
        
        # genai.Client() automatically finds and uses the GEMINI_API_KEY from the environment.
        try:
            self.client = genai.Client()
            logger.info(f"ScriptGenerator initialized successfully with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise RuntimeError("Could not initialize the Gemini client. Check API key and credentials.") from e
        
    def process(
        self,
        topics: List[str],
        keywords: List[str],
        tone: str = "informative and engaging",
        target_word_count: int = 300
    ) -> str | None:
        """
        Generates a complete video script.

        Returns:
            str: The generated script as a single formatted string, or None on failure.
        """
        # --- 1. Input Validation ---
        if not topics:
            logger.error("At least one topic must be provided to generate a script.")
            return None

        logger.info(f"Generating a ~{target_word_count}-word script with a '{tone}' tone...")
       
        instructional_prompt = f"""
        **ROLE:** You are an expert video scriptwriter for engaging online content.
        **TASK:** Write a complete, well-structured video script based on the provided details.

        **CONTEXT & INSTRUCTIONS:**
        1.  **Primary Topics to Cover:** {", ".join(topics)}
        2.  **Keywords to Include naturally:** {", ".join(keywords)}
        3.  **Desired Tone:** {tone}
        4.  **Target Length:** Approximately {target_word_count} words.
        5.  **Structure:** The script must have a clear introduction, body, and conclusion.
        6.  **HEADING FORMAT (CRITICAL):** All headings MUST be formatted like this: `:This is a Heading::`. Do not use markdown or any other format for headings.
        7.  **Content:** Use clear, concise language suitable for a voice-over. Break up long paragraphs.
        8.  **Output Format:** Provide ONLY the raw script content. Do not add a title, introduction, apologies, or any other conversational text outside of the script itself.

        **Generated Video Script:**
        """

        # --- 2. API Call with Retry Logic ---
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=instructional_prompt
            )
            
            # --- 3. Response Validation ---
            if response.text and response.text.strip():
                logger.info("Successfully generated video script.")
                return response.text.strip()
            else:
                # This handles cases where the API returns a success code but an empty response
                logger.warning("LLM returned an empty or invalid response. Aborting.")
                # We break here because retrying an empty response is unlikely to help.
                

        except Exception as e:
            logger.error(f"API call failed on attempt {e}")
            traceback.print_exc(file=sys.stderr)

        return None # Return None if all attempts fail or response is empty


# --- Independent Test Block ---
# This code only runs when you execute this file directly (e.g., `python src/processors/script_generator.py`)
if __name__ == '__main__':
    # --- Setup for Testing ---
    # Configure basic logging to see the output from the class
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Find the project's root directory to locate the .env file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(dotenv_path=os.path.join(project_root, '.env'))

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