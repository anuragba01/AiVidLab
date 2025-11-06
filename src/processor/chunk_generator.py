"""
Prompt Processor Module

This file contains the PromptProcessor class, a tool responsible for generating
a visually descriptive and effective prompt for a text-to-image AI.

Responsibilities:
- Take a segment of text, a high-level creative brief, and overall context.
- Use a Gemini Language Model (LLM) to synthesize these inputs.
- Generate a concise, artistic, and original prompt string tailored for
  an image generation model.
"""
import os
import time
import traceback

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("The 'google-generativeai' library is required. Please install it with 'pip install google-generativeai'")

class PromptProcessor:
    """
    Generates an image prompt by interpreting a text chunk within a creative context.
    """
    def __init__(self, api_key: str, model_name: str):
        """
        Initializes the PromptProcessor with a specific Gemini LLM.

        Args:
            api_key (str): The Google AI Studio API key.
            model_name (str): The specific Gemini LLM to use for prompt generation.
        """
        if not api_key:
            raise ValueError("API key must be provided for PromptProcessor.")
        if not model_name:
            raise ValueError("A model name must be provided for PromptProcessor.")

        print(f"Initializing PromptProcessor with model: {model_name}...")
        genai.configure(api_key=api_key)

        # Configure safety settings to be fairly permissive for creative text generation.
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        self.model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
        print("PromptProcessor initialized successfully.")

    def process(self, text_chunk: str, creative_brief: str, global_summary: str) -> str:
        """
        Generates a single, concise image prompt.

        Args:
            text_chunk (str): The specific text for the current scene or image.
            creative_brief (str): The high-level artistic direction (style, mood, constraints).
            global_summary (str): A summary of the entire script for thematic consistency.

        Returns:
            str: A generated image prompt. Returns a fallback prompt on failure.
        """
        if not text_chunk or not text_chunk.strip():
            print("Warning (PromptProcessor): Input text_chunk is empty. Returning fallback prompt.")
            return "An empty, neutral background."

        print(f"Generating image prompt for text: '{text_chunk[:70]}...'")

        # This is our "meta-prompt" or instruction template for the LLM.
        # It's engineered to guide the AI to produce the desired output format.
        instructional_prompt = f"""
        **ROLE:** You are an expert visual concept artist. Your task is to generate a single, concise, and visually descriptive prompt for a text-to-image AI.

        **CONTEXT:**
        1.  **Overall Script Theme:** "{global_summary}"
        2.  **Strict Creative Brief (Style, Mood, Constraints):** "{creative_brief}"
        3.  **Specific Text for This Image:** "{text_chunk}"

        **INSTRUCTIONS:**
        1.  **Synthesize:** Read all context. Your primary focus is the 'Specific Text for This Image', but the style must strictly adhere to the 'Strict Creative Brief' and the theme must align with the 'Overall Script Theme'.
        2.  **Be Visual:** Use strong nouns, evocative adjectives, and describe lighting, composition, and mood. Do not describe abstract concepts; describe what to *see*.
        3.  **Be Concise:** The entire prompt should be a single paragraph of 30-70 words.
        4.  **Output Format:** PROVIDE ONLY THE PROMPT ITSELF. Do not add explanations, labels, or apologies. Do not write "Prompt:" or repeat the instructions.

        **Generated Visual Prompt:**
        """

        try:
            # Configure the generation parameters for creative output.
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=150,
                temperature=0.8, # Higher temperature for more creative/varied prompts
            )

            response = self.model.generate_content(
                instructional_prompt,
                generation_config=generation_config
            )

            if response.text and response.text.strip():
                generated_prompt = response.text.strip()
                print(f"  - Generated Prompt: \"{generated_prompt}\"")
                return generated_prompt
            else:
                print("Warning (PromptProcessor): LLM returned an empty response. Using fallback.")
                return f"A cinematic and visually striking scene representing the concept: {text_chunk}"

        except Exception as e:
            print(f"ERROR (PromptProcessor): An exception occurred during the API call: {e}")
            traceback.print_exc()
            # Provide a safe but useful fallback prompt.
            return f"A detailed and emotionally resonant visual for the text: {text_chunk}"

# --- Independent Test Block ---
if __name__ == '__main__':
    print("\n--- Running Independent Test for PromptProcessor ---")

    test_api_key = os.getenv("GEMINI_API_KEY", "PASTE_YOUR_API_KEY_HERE_FOR_TESTING")

    if "PASTE_YOUR" in test_api_key:
        print("\nWARNING: Please set the GEMINI_API_KEY environment variable for testing.")
        print("Skipping live API test.")
    else:
        # These would come from config.json
        TEST_MODEL_NAME = "gemini-1.5-flash-latest"

        # These would come from user_input.json
        TEST_CHUNK = "It is in the act of beginning that the true power lies, not in the contemplation of the destination."
        TEST_BRIEF = "Style: Ancient philosophy scroll, ink wash painting (sumi-e), minimalist. Avoid: Modern objects, people's faces, bright colors."
        TEST_SUMMARY = "The video is about the importance of taking the first step on a long journey."

        try:
            # 1. Instantiate the tool
            prompt_tool = PromptProcessor(api_key=test_api_key, model_name=TEST_MODEL_NAME)

            # 2. Use the tool to process the data
            final_prompt = prompt_tool.process(
                text_chunk=TEST_CHUNK,
                creative_brief=TEST_BRIEF,
                global_summary=TEST_SUMMARY
            )

            # 3. Verify the output
            if final_prompt:
                print("\nSUCCESS: Prompt generation completed.")
                print(f"  - Final Generated Prompt: \"{final_prompt}\"")
            else:
                print("\nFAILURE: The processor returned an empty or invalid prompt.")

        except (ValueError, RuntimeError, ImportError) as e:
            print(f"\nFAILURE: The test failed with an error: {e}")