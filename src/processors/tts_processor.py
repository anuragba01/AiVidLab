"""
Text-to-Speech (TTS) Processor Module

Converts text into audio using the Google Gemini TTS model.
Production-ready version with structured logging and clean error handling.
"""

import os
import time
import logging
from google import genai
from google.generativeai import types

logger = logging.getLogger(__name__)


class TTSProcessor:
    """Handles text-to-speech synthesis using the Gemini API."""

    def __init__(self, model_name: str):
        if not model_name:
            raise ValueError("A model name must be provided for TTSProcessor.")
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")
        
        self.model_name = model_name
        logger.info(f"TTSProcessor initialized with model: {model_name}")

    def process(self, text_to_speak: str, voice_name: str, api_timeout_s: int = 300) -> bytes:
        """Generates audio bytes for the given text using the specified voice."""
        if not text_to_speak or not text_to_speak.strip():
            logger.warning("Empty text_to_speak provided; returning empty bytes.")
            return b""

        start_time = time.time()
        try:
            client = genai.Client()
            generation_config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=text_to_speak,
                generation_config=generation_config,
                request_options={"timeout": api_timeout_s},
            )

            duration = time.time() - start_time
            logger.info(f"TTS synthesis completed in {duration:.2f}s using voice '{voice_name}'.")

            try:
                audio_bytes = (
                    response.candidates[0]
                    .content.parts[0]
                    .inline_data.data
                )
            except (AttributeError, IndexError):
                audio_bytes = None

            if audio_bytes:
                logger.info(f"Generated {len(audio_bytes)} bytes of audio.")
                return audio_bytes

            block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", "Unknown")
            raise ValueError(f"API response missing audio data. Block reason: {block_reason}")

        except Exception as e:
            logger.exception("TTSProcessor failed during API call.")
            raise RuntimeError(f"TTS synthesis failed: {e}") from e


# --- Optional Test Block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    TEST_MODEL = "gemini-2.5-flash-preview-tts"
    TEST_VOICE = "Kore"
    TEST_TEXT = "This is a production test of the TTS Processor module."

    tts = TTSProcessor(model_name=TEST_MODEL)
    audio_bytes = tts.process(TEST_TEXT, TEST_VOICE)

    if audio_bytes:
        output = "test_output.wav"
        with open(output, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"Audio file saved as {output} ({len(audio_bytes)} bytes).")
    else:
        logger.warning("No audio data returned.")
