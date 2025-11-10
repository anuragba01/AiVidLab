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
import traceback
from google import genai
from google.genai import types

class PromptProcessor:
    """
    Generates an image prompt by interpreting a text chunk within a creative context.
    """
    def __init__(self, model_name: str):
        """
        Initializes the PromptProcessor with a specific Gemini LLM.
        The API key is automatically sourced from the GEMINI_API_KEY environment variable.

        Args:
            model_name (str): The specific Gemini LLM to use for prompt generation.
        """
        if not model_name:
            raise ValueError("A model name must be provided for PromptProcessor.")

        print(f"Initializing PromptProcessor with model: {model_name}...")
        self.model_name = model_name
        # The genai.Client() will be instantiated in the process method.
        print("PromptProcessor initialized successfully.")
        
        
    # --- Replace the existing process method with this corrected version ---
    def process(self, text_chunk: str, creative_brief: str, global_summary: str) -> str:
        """
        Generates a single, concise image prompt.
        """
        if not text_chunk or not text_chunk.strip():
            print("Warning (PromptProcessor): Input text_chunk is empty. Returning fallback prompt.")
            return "An empty, neutral background."

        print(f"Generating image prompt for text: '{text_chunk[:70]}...'")

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
            client = genai.Client()
            
            # This is the corrected configuration using the 'generation_config' parameter
            generation_config = types.GenerateContentConfig(
                temperature=0.8,
                safety_settings=[
                    {"category": c, "threshold": types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE}
                    for c in types.HarmCategory if c != types.HarmCategory.HARM_CATEGORY_UNSPECIFIED
                ]
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=instructional_prompt,
                generation_config=generation_config # Correct parameter name
            )

            if (response.candidates and response.candidates[0].content and
                response.candidates[0].content.parts and
                response.candidates[0].content.parts[0].text):
                
                generated_prompt = response.candidates[0].content.parts[0].text.strip()
                print(f"  - Generated Prompt: \"{generated_prompt}\"")
                return generated_prompt
            else:
                print("Warning (PromptProcessor): LLM returned an empty response. Using fallback.")
                return f"A cinematic and visually striking scene representing the concept: {text_chunk}"

        except Exception as e:
            print(f"ERROR (PromptProcessor): An exception occurred during the API call: {e}")
            # Provide a safe but useful fallback prompt.
            return f"A detailed and emotionally resonant visual for the text: {text_chunk}"

    
# --- Independent Test Block ---
if __name__ == '__main__':
    print("\n--- Running Independent Test for PromptProcessor ---")

    # The genai.Client() automatically looks for the GEMINI_API_KEY environment variable.
    if not os.getenv("GEMINI_API_KEY"):
        print("\nFATAL ERROR: Please set the GEMINI_API_KEY environment variable for testing.")
        print("Skipping live API test.")
    else:
        # These settings would normally come from your configuration files.
        TEST_MODEL_NAME = "gemini-2.0-flash-lite" # A great model for this kind of synthesis task.

        # Sample inputs for the test
        TEST_GLOBAL_SUMMARY = "A short video about the philosophy of Stoicism, focusing on emotional resilience and inner peace."
        TEST_CREATIVE_BRIEF = "Cinematic, dramatic lighting, photorealistic, muted color palette, thoughtful and serene mood. Use a shallow depth of field."
        TEST_TEXT_CHUNK = "The true measure of a man is not how he behaves in moments of comfort, but how he stands at times of challenge and controversy."

        try:
            # 1. Instantiate the processor tool (no API key needed here)
            prompt_tool = PromptProcessor(model_name=TEST_MODEL_NAME)
            
            # 2. Use the tool to generate a prompt
            final_prompt = prompt_tool.process(
                text_chunk=TEST_TEXT_CHUNK,
                creative_brief=TEST_CREATIVE_BRIEF,
                global_summary=TEST_GLOBAL_SUMMARY
            )

            # 3. Verify and print the output
            if final_prompt and "background" not in final_prompt and "concept" not in final_prompt:
                print("\n--- SUCCESS: Prompt Generation Complete ---")
                print("\n--- Generated Prompt ---")
                print(final_prompt)
                print("------------------------")
            else:
                print("\nFAILURE: The processor returned a fallback prompt or was empty.")

        except (ValueError, RuntimeError, ImportError) as e:
            print(f"\nFAILURE: The test failed with an error: {e}")