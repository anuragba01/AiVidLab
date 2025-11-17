"""
Text-to-Speech (TTS) Processor Module

Converts text into audio using the Google Gemini TTS model.
Production-ready version with structured logging and clean error handling.
"""

import os
import time
import logging
from google import genai
from google.genai import types
import wave

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

    def process(self, text_to_speak: str, voice_name: str) :
        """Generates audio bytes for the given text using the specified voice."""
        if not text_to_speak or not text_to_speak.strip():
            logger.warning("Empty text_to_speak provided; returning empty bytes.")
            return b""
        
        start_time = time.time()
        try:
            # The original code had 'genai.Client()', which is now part of the standard library.
            # No change is needed here as it aligns with the provided snippet.
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
                config=generation_config,
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


    def wave_file(self,filename, pcm, channels=1, rate=24000, sample_width=2):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)


