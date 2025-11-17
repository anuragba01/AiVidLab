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
        # 1. Input Validation 
        if not topics:
            logger.error("At least one topic must be provided to generate a script.")
            return None

        logger.info(f"Generating a ~{target_word_count}-word script with a '{tone}' tone...")
       
        instructional_prompt = f"""
        Create a  cinematic, philosophical YouTube video script written in a reflective, narrative tone. The script should feel immersive, emotional, and thought-provoking, with smooth transitions and a strong sense of storytelling. Let the ideas unfold naturally through introspection, metaphors, and real-life observations. Keep the pacing calm and engaging, almost like a philosophical monologue meant to make the listener pause and think.
        Do not include any headings, topic titles, or numbered sections anywhere in the script. The entire output should read as one continuous flow of ideas, expressed with clarity, depth, and elegance. Remember- in response only should be actual script and non even a word or symbol other than script since it is use directly for text generation.

        **CONTEXT & INSTRUCTIONS:**
        1.  **Primary Topics to Cover:** {", ".join(topics)}
        2.  **Keywords to Include naturally:** {", ".join(keywords)}
        3.  **Desired Tone:** {tone}
        4.  **Target Length:** Approximately {target_word_count} words. must adhare the word count , this is necessory.
        8.  **Output Format:** Provide ONLY the raw script content. Do not add a title, introduction, apologies, or any other conversational text outside of the script itself.

        **Generated Video Script:**
        """

        # 2. API Call with Retry Logic 
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=instructional_prompt
            )
            
            # 3. Response Validation
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


