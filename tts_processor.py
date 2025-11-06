"""
Text-to-Speech (TTS) Processor Module

This file contains the TTSProcessor class, a self-contained tool responsible
for converting a given string of text into audio data using the Google Gemini API.

Responsibilities:
- Initialize and configure the Gemini client for text-to-speech.
- Encapsulate the logic for the API call.
- Handle API responses and errors robustly.
- Return raw audio data as bytes.
"""
import os
import time
import traceback

try:
    import google.generativeai as genai
    from google.generativeai import types as genai_types
except ImportError:
    raise ImportError("The 'google-generativeai' library is required. Please install it with 'pip install google-generativeai'")

class TTSProcessor:
    """
    A processor to handle Text-to-Speech synthesis using the Gemini API.
    """
    def __init__(self, api_key: str, model_name: str):
        """
        Initializes the TTSProcessor by configuring the Gemini client.

        Args:
            api_key (str): The Google AI Studio API key.
            model_name (str): The specific Gemini model to use for TTS.
        """
        if not api_key:
            raise ValueError("API key must be provided for TTSProcessor.")
        if not model_name:
            raise ValueError("A model name must be provided for TTSProcessor.")
            
        print(f"Initializing TTSProcessor with model: {model_name}...")
        
        # Configure the genai library. It's safe to call this multiple times
        # if other processors also use it.
        genai.configure(api_key=api_key)
        
        # Define safety settings to be permissive for TTS content, as we are not
        # expecting harmful content in our scripts.
        safety_settings = [
            {"category": c, "threshold": genai_types.HarmBlockThreshold.BLOCK_NONE}
            for c in genai_types.HarmCategory if c != genai_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED
        ]
        
        # Create the generative model instance.
        self.model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
        print("TTSProcessor initialized successfully.")

    def process(self, text_to_speak: str, voice_name: str, api_timeout_s: int = 300) -> bytes:
        """
        Processes the input text to generate speech and returns the raw audio data.

        Args:
            text_to_speak (str): The text content to be synthesized.
            voice_name (str): The name of the prebuilt voice to use (e.g., 'en-US-Standard-C').
            api_timeout_s (int): The timeout in seconds for the API request.

        Returns:
            bytes: The raw WAV audio data. Returns an empty bytes object if input text is empty.

        Raises:
            ValueError: If the API response does not contain valid audio data.
            RuntimeError: If there is a critical exception during the API call.
        """
        if not text_to_speak or not text_to_speak.strip():
            print("Warning (TTSProcessor): Input text is empty. Returning empty bytes.")
            return b''

        print(f"Synthesizing audio using voice '{voice_name}' for text: '{text_to_speak[:70]}...'")
        
        try:
            start_time = time.time()
            
            # This payload structure is specific to the Gemini TTS API.
            generation_config_payload = {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": voice_name}
                    }
                },
                "candidate_count": 1
            }

            response = self.model.generate_content(
               contents=text_to_speak,
               generation_config=generation_config_payload,
               request_options={"timeout": api_timeout_s}
            )
            
            duration = time.time() - start_time
            print(f"Gemini TTS API call completed in {duration:.2f} seconds.")

            # --- Robustly check the response for the audio data ---
            if (response.candidates and response.candidates[0].content and
                response.candidates[0].content.parts and
                response.candidates[0].content.parts[0].inline_data and
                hasattr(response.candidates[0].content.parts[0].inline_data, 'data') and
                response.candidates[0].content.parts[0].inline_data.data):
                
                audio_bytes = response.candidates[0].content.parts[0].inline_data.data
                print(f"Successfully received {len(audio_bytes)} bytes of audio data.")
                return audio_bytes
            else:
                block_reason = "Unknown reason."
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    block_reason = response.prompt_feedback.block_reason
                raise ValueError(f"Gemini TTS API response did not contain valid audio data. Block reason: {block_reason}")

        except Exception as e:
            print(f"ERROR (TTSProcessor): An exception occurred during the API call: {e}")
            traceback.print_exc()
            # Re-raise as a RuntimeError to signal a critical failure in this step.
            raise RuntimeError(f"Failed to generate speech from text. Original error: {e}") from e


# --- Independent Test Block ---
# This allows you to run `python processors/tts_processor.py` to test this file in isolation.
if __name__ == '__main__':
    print("\n--- Running Independent Test for TTSProcessor ---")
    
    # In a real environment, use `os.getenv("GEMINI_API_KEY")`
    # For this test, you might need to paste your key here.
    # It's better to set it as an environment variable in your terminal before running.
    test_api_key = os.getenv("GEMINI_API_KEY", "PASTE_YOUR_API_KEY_HERE_FOR_TESTING")
    
    if "PASTE_YOUR" in test_api_key:
        print("\nWARNING: Please set the GEMINI_API_KEY environment variable or paste it directly in the script for testing.")
        print("Skipping live API test.")
    else:
        # These settings would normally come from your config.json
        TEST_MODEL_NAME = "models/text-to-speech"
        TEST_VOICE_NAME = "en-US-Standard-C"
        
        # This text would come from your user_input.json
        TEST_SCRIPT_TEXT = "This is a live test of the text-to-speech processor. If you can hear this, it means the tool is working correctly."

        try:
            # 1. Instantiate the processor tool
            tts_tool = TTSProcessor(api_key=test_api_key, model_name=TEST_MODEL_NAME)
            
            # 2. Use the tool to process the data
            generated_audio_bytes = tts_tool.process(
                text_to_speak=TEST_SCRIPT_TEXT,
                voice_name=TEST_VOICE_NAME
            )
            
            # 3. Save the output to verify it's correct
            if generated_audio_bytes:
                output_filename = "test_tts_output.wav"
                with open(output_filename, "wb") as f:
                    f.write(generated_audio_bytes)
                print(f"\nSUCCESS: Test audio file saved to '{output_filename}'")
                print("You can now play this file to verify the output.")
            else:
                print("\nFAILURE: The processor returned empty bytes, but no error was raised.")

        except (ValueError, RuntimeError, ImportError) as e:
            print(f"\nFAILURE: The test failed with an error: {e}")