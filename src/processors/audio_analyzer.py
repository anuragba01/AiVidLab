"""
Audio Analyzer Processor Module

This file contains the AudioAnalyzer class, a tool responsible for processing
raw audio data to extract meaningful information for video creation.

Responsibilities:
- Transcribe audio using a speech-to-text model (Whisper).
- From the transcription, provide two key outputs:
  1. A list of pacing chunks (based on silence) for image/scene timing.
  2. A detailed list of all words with their precise timestamps for subtitles.
- Perform this analysis once, efficiently providing data for multiple downstream tasks.
"""
import os
import io
import tempfile
import traceback
from typing import List, Dict, Any
from pydub import AudioSegment

# Pydub is excellent for audio manipulation and silence detection.
try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
except ImportError:
    raise ImportError("The 'pydub' library is required. Please install it with 'pip install pydub'")

# Whisper-timestamped provides the word-level timestamp data.
try:
    import whisper_timestamped as whisper
except ImportError:
    raise ImportError("The 'whisper-timestamped' library is required. Please install it with 'pip install git+https://github.com/linto-ai/whisper-timestamped.git'")

class AudioAnalyzer:
    """
    Analyzes audio data to produce pacing chunks and word-level timestamps.
    """
    def __init__(self, model_size: str = "small", device: str = None):
        """
        Initializes the AudioAnalyzer by loading the Whisper STT model.

        Args:
            model_size (str): The size of the Whisper model to load
                              ("tiny", "base", "small", "medium", "large").
            device (str, optional): The device to run the model on ("cuda", "cpu").
                                    If None, it will auto-detect.
        """
        print(f"Initializing AudioAnalyzer by loading Whisper model '{model_size}'...")
        self.model_size = model_size
        self.device = device
        self.model = whisper.load_model(self.model_size, device=self.device)
        print("AudioAnalyzer initialized successfully.")

    def process(self, audio_bytes: bytes, chunk_config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Processes audio bytes to extract pacing and transcription data.

        Args:
            audio_bytes (bytes): The raw WAV audio data.
            chunk_config (Dict[str, Any]): A dictionary of parameters for chunking, e.g.,
                                           {'min_silence_len_ms': 500, 'silence_thresh_offset': -10}.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing two keys:
                'pacing_chunks': List of {'raw_text': str, 'duration_ms': int}.
                'word_timestamps': List of {'text': str, 'start': float, 'end': float}.
        """
        if not audio_bytes:
            print("Warning (AudioAnalyzer): Input audio_bytes is empty. Returning empty results.")
            return {'pacing_chunks': [], 'word_timestamps': []}

        print("Analyzing audio: Transcribing and identifying pacing chunks...")
        
        # Whisper works best with files, so we use a temporary file.
        # `tempfile.NamedTemporaryFile` handles cleanup automatically.
        try:
            # Create a file-like object in memory from the raw bytes
            audio_stream = io.BytesIO(audio_bytes)
            # Pydub will auto-detect the format (e.g., MP3) and decode it
            audio_segment = AudioSegment.from_file(audio_stream)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                audio_segment.export(tmpfile.name, format="wav")
                tmpfile.write(audio_bytes)
                tmpfile.flush() # Ensure all data is written to disk
                
                # --- 1. Perform the full transcription to get word timestamps ---
                audio_for_ts = whisper.load_audio(tmpfile.name)
                result = whisper.transcribe(self.model, audio_for_ts, language="en")
                
                word_timestamps = self._extract_word_timestamps(result)

                # --- 2. Use Pydub for silence-based chunking ---
                # Pydub needs to load the audio from the file as well.
                audio_segment = AudioSegment.from_file(tmpfile.name, format="wav")
                silence_chunks = self._chunk_on_silence(audio_segment, chunk_config)
                
                # --- 3. Align transcription text with silence chunks ---
                pacing_chunks = self._align_text_to_chunks(silence_chunks, word_timestamps)

                print(f"Analysis complete. Found {len(pacing_chunks)} pacing chunks and {len(word_timestamps)} words.")
                return {
                    'pacing_chunks': pacing_chunks,
                    'word_timestamps': word_timestamps
                }

        except Exception as e:
            print(f"ERROR (AudioAnalyzer): An exception occurred during audio processing: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to analyze audio. Original error: {e}") from e

    def _extract_word_timestamps(self, transcription_result: Dict) -> List[Dict[str, Any]]:
        """Helper to flatten the Whisper result into a simple list of words."""
        all_words = []
        for segment in transcription_result.get("segments", []):
            for word in segment.get("words", []):
                all_words.append({
                    'text': word['text'],
                    'start': word['start'],
                    'end': word['end']
                })
        return all_words

    def _chunk_on_silence(self, audio_segment: AudioSegment, config: Dict) -> List[AudioSegment]:
        """Helper to split audio based on silence using Pydub."""
        # Use a dynamic silence threshold based on the audio's energy
        silence_thresh = audio_segment.dBFS + config.get('silence_thresh_offset', -16)
        
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=config.get('min_silence_len_ms', 500),
            silence_thresh=silence_thresh,
            keep_silence=250 # Keep a bit of silence for natural pauses
        )
        return chunks if chunks else [audio_segment] # If no silence, return the whole audio as one chunk

    def _align_text_to_chunks(self, audio_chunks: List[AudioSegment], words: List[Dict]) -> List[Dict]:
        """Helper to map the transcribed words to their respective audio chunks."""
        pacing_chunks = []
        word_idx = 0
        current_time_ms = 0
        
        for chunk in audio_chunks:
            chunk_duration_ms = len(chunk)
            chunk_end_time_s = (current_time_ms + chunk_duration_ms) / 1000.0
            
            chunk_text_parts = []
            start_word_idx = word_idx
            
            # Find all words that fall within this chunk's time range
            while word_idx < len(words) and words[word_idx]['start'] < chunk_end_time_s:
                chunk_text_parts.append(words[word_idx]['text'])
                word_idx += 1
            
            # Reconstruct the text for this chunk
            if chunk_text_parts:
                pacing_chunks.append({
                    'raw_text': " ".join(chunk_text_parts),
                    'duration_ms': chunk_duration_ms
                })
            
            current_time_ms += chunk_duration_ms
            
        return pacing_chunks

# --- Independent Test Block ---
if __name__ == '__main__':
    print("\n--- Running Independent Test for AudioAnalyzer ---")
    
    # This test requires a sample audio file to exist.
    # We will use the output of the tts_processor test.
    test_audio_file = "test_tts_output.wav"
    
    if not os.path.exists(test_audio_file):
        print(f"\nWARNING: Test audio file '{test_audio_file}' not found.")
        print("Please run the `tts_processor.py` test first to generate it.")
    else:
        # These would come from config.json
        TEST_MODEL_SIZE = "tiny" # Use tiny for a fast test
        TEST_CHUNK_CONFIG = {
            "min_silence_len_ms": 400,
            "silence_thresh_offset": -16
        }
        
        try:
            with open(test_audio_file, "rb") as f:
                audio_data = f.read()

            # 1. Instantiate the analyzer
            analyzer_tool = AudioAnalyzer(model_size=TEST_MODEL_SIZE)

            # 2. Process the audio data
            analysis_result = analyzer_tool.process(audio_data, TEST_CHUNK_CONFIG)

            # 3. Verify the output
            if analysis_result and analysis_result['word_timestamps'] and analysis_result['pacing_chunks']:
                print("\nSUCCESS: Audio analysis completed.")
                print(f"  - Found {len(analysis_result['pacing_chunks'])} pacing chunks.")
                print(f"  - Transcribed {len(analysis_result['word_timestamps'])} words.")
                
                print("\n--- Sample Pacing Chunk (First Chunk) ---")
                print(analysis_result['pacing_chunks'][0])
                
                print("\n--- Sample Word Timestamps (First 5 Words) ---")
                for i, word in enumerate(analysis_result['word_timestamps'][:5]):
                    print(f"  {i+1}: {word}")
            else:
                print("\nFAILURE: The analysis result was empty.")

        except (RuntimeError, ImportError) as e:
            print(f"\nFAILURE: The test failed with an error: {e}")