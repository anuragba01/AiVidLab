"""
Text-to-Speech (TTS) Processor Module (Refactored)

This file contains the TTSProcessor class, a self-contained tool responsible
for converting a given string of text into audio data using the Google Gemini API.
This version uses the modern `genai.Client()` pattern.
"""
import os
import time
import traceback
from google import genai
from google.generativeai import types

class TTSProcessor:
    """
    A processor to handle Text-to-Speech synthesis using the Gemini API.
    """
    def __init__(self, model_name: str):
        """
        Initializes the TTSProcessor. The API key is sourced automatically from
        the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.

        Args:
            model_name (str): The specific Gemini model to use for TTS.
        """
        if not model_name:
            raise ValueError("A model name must be provided for TTSProcessor.")
            
        print(f"Initializing TTSProcessor with model: {model_name}...")
        self.model_name = model_name
        # The client will be instantiated on-demand in the process method.
        print("TTSProcessor initialized successfully.")

    def process(self, text_to_speak: str, voice_name: str, api_timeout_s: int = 300) -> bytes:
        """
        Processes the input text to generate speech and returns the raw audio data.

        Args:
            text_to_speak (str): The text content to be synthesized.
            voice_name (str): The name of the prebuilt voice to use (e.g., 'Kore', 'en-US-Standard-C').
            api_timeout_s (int): The timeout in seconds for the API request.

        Returns:
            bytes: The raw WAV audio data. Returns an empty bytes object if input text is empty.
        """
        if not text_to_speak or not text_to_speak.strip():
            print("Warning (TTSProcessor): Input text is empty. Returning empty bytes.")
            return b''

        print(f"Synthesizing audio using voice '{voice_name}' for text: '{text_to_speak[:70]}...'")
        
        try:
            start_time = time.time()
            client = genai.Client()
            
            # This structured payload is built using the types module, as per the new config.
            generation_config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                           voice_name=voice_name,
                        )
                    )
                ),
            )

            # The API call now uses the client.models.generate_content method.
            response = client.models.generate_content(
               model=self.model_name,
               contents=text_to_speak,
               generation_config=generation_config,
               request_options={"timeout": api_timeout_s}
            )
            
            duration = time.time() - start_time
            print(f"Gemini TTS API call completed in {duration:.2f} seconds.")

            # Robustly check the response for the audio data
            if (response.candidates and response.candidates[0].content and
                response.candidates[0].content.parts and
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
            raise RuntimeError(f"Failed to generate speech from text. Original error: {e}") from e


# --- Independent Test Block (Updated for the new class structure) ---
if __name__ == '__main__':
    print("\n--- Running Independent Test for TTSProcessor ---")
    
    # The genai.Client() automatically finds the key from the environment.
    # This check ensures the user has set it.
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("\nFATAL ERROR: Please set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        print("Skipping live API test.")
    else:
        # Use the correct model and voice names from your working config.
        TEST_MODEL_NAME = "gemini-2.5-flash-preview-tts"
        TEST_VOICE_NAME = "Kore"
        
        TEST_SCRIPT_TEXT = "This is a live test of the text-to-speech processor. If you can hear this, it means the tool is working correctly."

        try:
            # 1. Instantiate the tool (no longer needs an api_key).
            tts_tool = TTSProcessor(model_name=TEST_MODEL_NAME)
            
            # 2. Use the tool to process the data.
            generated_audio_bytes = tts_tool.process(
                text_to_speak=TEST_SCRIPT_TEXT,
                voice_name=TEST_VOICE_NAME
            )
            
            # 3. Save the output to verify it's correct.
            if generated_audio_bytes:
                output_filename = "test_tts_output.wav"
                with open(output_filename, "wb") as f:
                    f.write(generated_audio_bytes)
                
                file_size = os.path.getsize(output_filename)
                print(f"\nSUCCESS: Test audio file saved to '{output_filename}' ({file_size} bytes)")
                print("You can now play this file to verify the output.")
            else:
                print("\nFAILURE: The processor returned empty bytes, but no error was raised.")

        except (ValueError, RuntimeError, ImportError) as e:
            print(f"\nFAILURE: The test failed with an error: {e}")