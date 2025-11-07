"""
Script Generator Processor Module

This file contains the ScriptGenerator class, a tool responsible for creating
a complete video script from a set of topics and keywords.

Responsibilities:
- Take high-level concepts (topics, keywords) and constraints (tone, length).
- Use a Gemini Language Model (LLM) to expand these concepts into a coherent,
  well-structured video script.
- Format the script with clear headings (using the ':Heading Text::' syntax)
  that can be used by the SubtitleProcessor and for YouTube chapters.
- Return the final script as a formatted string.
"""
import os
import traceback
from typing import List

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("The 'google-generativeai' library is required. Please install it with 'pip install google-generativeai'")

class ScriptGenerator:
    """
    Generates a video script using a Gemini LLM based on user-provided topics.
    """
    def __init__(self, api_key: str, model_name: str):
        """
        Initializes the ScriptGenerator with a specific Gemini LLM.

        Args:
            api_key (str): The Google AI Studio API key.
            model_name (str): The specific Gemini LLM to use for script generation.
        """
        if not api_key:
            raise ValueError("API key must be provided for ScriptGenerator.")
        if not model_name:
            raise ValueError("A model name must be provided for ScriptGenerator.")

        print(f"Initializing ScriptGenerator with model: {model_name}...")
        genai.configure(api_key=api_key)

        # Configure safety settings appropriate for creative writing.
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        self.model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
        print("ScriptGenerator initialized successfully.")

    def process(
        self,
        topics: List[str],
        keywords: List[str],
        tone: str = "informative and engaging",
        target_word_count: int = 300
    ) -> str:
        """
        Generates a complete video script.

        Args:
            topics (List[str]): A list of the main topics the script should cover.
            keywords (List[str]): A list of specific keywords to include.
            tone (str): The desired tone of the script (e.g., "inspirational",
                        "humorous", "formal").
            target_word_count (int): The approximate desired length of the script.

        Returns:
            str: The generated script as a single formatted string. Returns None on failure.
        """
        if not topics:
            raise ValueError("At least one topic must be provided to generate a script.")

        print(f"Generating a ~{target_word_count}-word script with a '{tone}' tone...")
        
        # This is our prompt engineering for the scriptwriter AI.
        # It is highly structured to get a consistent and high-quality output.
        instructional_prompt = f"""
        **ROLE:** You are an expert video scriptwriter. Your task is to write a complete, engaging, and well-structured video script based on the provided details.

        **CONTEXT & INSTRUCTIONS:**
        1.  **Primary Topics to Cover:** {", ".join(topics)}
        2.  **Keywords to Include:** {", ".join(keywords)}
        3.  **Desired Tone:** {tone}
        4.  **Target Length:** Approximately {target_word_count} words.
        5.  **Structure:** The script must be logically structured with an introduction, a body that covers the topics, and a conclusion.
        6.  **HEADING FORMAT:** You MUST format all headings by enclosing them in a single colon on the left and a double colon on the right. For example: `:This is a Heading::`. This is critical for the video pipeline. Do not use any other heading format like markdown.
        7.  **Content:** Write in clear, concise language suitable for being spoken as a voice-over. Break up long paragraphs into shorter, more digestible sentences.
        8.  **Output Format:** Provide ONLY the script itself. Do not include a title, introduction, or any explanatory text outside of the script content.

        **Generated Video Script:**
        """

        try:
            # Configure the generation for a creative, long-form text task.
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=2048, # Allow for longer scripts
                temperature=0.75,     # A good balance of creativity and coherence
            )

            response = self.model.generate_content(
                instructional_prompt,
                generation_config=generation_config
            )

            if response.text and response.text.strip():
                generated_script = response.text.strip()
                print("Successfully generated video script.")
                return generated_script
            else:
                print("Warning (ScriptGenerator): LLM returned an empty response.")
                return None

        except Exception as e:
            print(f"ERROR (ScriptGenerator): An exception occurred during the API call: {e}")
            traceback.print_exc()
            return None

# --- Independent Test Block ---
if __name__ == '__main__':
    print("\n--- Running Independent Test for ScriptGenerator ---")
    
    test_api_key = os.getenv("GEMINI_API_KEY", "PASTE_YOUR_API_KEY_HERE_FOR_TESTING")

    if "PASTE_YOUR" in test_api_key:
        print("\nWARNING: Please set the GEMINI_API_KEY environment variable for testing.")
        print("Skipping live API test.")
    else:
        # These would come from a user or another program
        TEST_TOPICS = ["The philosophy of Stoicism", "Practical applications in modern life"]
        TEST_KEYWORDS = ["Marcus Aurelius", "emotional resilience", "mindfulness", "virtue"]
        TEST_TONE = "calm, inspirational, and educational"
        TEST_LENGTH = 250

        try:
            # 1. Instantiate the tool
            script_tool = ScriptGenerator(
                api_key=test_api_key,
                model_name="gemini-1.5-flash-latest" # Good model for this task
            )

            # 2. Use the tool to generate the script
            final_script = script_tool.process(
                topics=TEST_TOPICS,
                keywords=TEST_KEYWORDS,
                tone=TEST_TONE,
                target_word_count=TEST_LENGTH
            )

            # 3. Verify and print the output
            if final_script:
                print("\n--- SUCCESS: Script Generation Complete ---")
                
                # Save the script to a file to simulate the workflow
                output_filename = "test_generated_script.txt"
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(final_script)
                
                print(f"  - Test script saved to '{output_filename}'")
                print("\n--- Generated Script ---")
                print(final_script)
                print("------------------------")
                # Check if the critical heading format was used
                if ":" in final_script and "::" in final_script:
                    print("\nVerification PASSED: Script contains the required ':Heading::' format.")
                else:
                    print("\nVerification FAILED: Script may be missing the required ':Heading::' format.")
            else:
                print("\nFAILURE: The processor returned None, indicating an error.")

        except (ValueError, RuntimeError, ImportError) as e:
            print(f"\nFAILURE: The test failed with an error: {e}")