import io
import traceback
from typing import List, Dict, Any
import numpy as np # Add this import

try:
    from pydub import AudioSegment
except ImportError:
    # We still use pydub for initial loading, but not for chunking
    raise ImportError("Install 'pydub' with: pip install pydub")

try:
    import whisper_timestamped as whisper
except ImportError:
    raise ImportError("Install 'whisper-timestamped' with: pip install git+https://github.com/linto-ai/whisper-timestamped.git")


class AudioAnalyzer:
    """Analyzes audio data to produce pacing chunks and word-level timestamps."""

    def __init__(self, model_size: str = "small", device: str = None):
        self.model_size = model_size
        self.device = device
        self.model = whisper.load_model(model_size, device=device)

    def process(self, audio_bytes: bytes, chunk_config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Processes audio bytes to extract pacing and transcription data."""
        if not audio_bytes:
            return {'pacing_chunks': [], 'word_timestamps': []}

        try:
            # 1. Load audio from memory into a NumPy array (avoids temp files)
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            # Convert to mono and the correct sample rate for Whisper
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
            audio_np = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0

            # 2. Transcribe ONCE to get the single source of truth
            result = whisper.transcribe(self.model, audio_np, language="en")
            
            # 3. Extract word timestamps from the result
            word_timestamps = self._extract_word_timestamps(result)

            # 4. Build pacing chunks directly from the word timestamps.
            # The chunk_config may contain:
            # - min_silence_len_ms: silence gap (ms) that separates chunks
            # - min_chunk_duration_s: minimum desired chunk duration (s)
            # - max_chunk_duration_s: maximum desired chunk duration (s)
            min_silence_ms = int(chunk_config.get('min_silence_len_ms', 500))
            min_chunk_ms = int(float(chunk_config.get('min_chunk_duration_s', 0)) * 1000)
            max_chunk_ms = int(float(chunk_config.get('max_chunk_duration_s', 0)) * 1000)

            pacing_chunks = self._create_pacing_chunks_from_words(
                word_timestamps,
                min_silence_ms,
                min_chunk_ms,
                max_chunk_ms
            )

            return {
                'pacing_chunks': pacing_chunks,
                'word_timestamps': word_timestamps
            }

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Audio analysis failed: {e}") from e

    def _extract_word_timestamps(self, transcription_result: Dict) -> List[Dict[str, Any]]:
        """Flattens Whisper result into a list of words."""
        all_words = []
        if "segments" in transcription_result:
            for seg in transcription_result["segments"]:
                if "words" in seg:
                    all_words.extend(seg["words"])
        # A simpler way to format, ensuring 'text', 'start', 'end' are present
        return [{'text': w['text'], 'start': w['start'], 'end': w['end']} for w in all_words]

    def _create_pacing_chunks_from_words(
        self,
        words: List[Dict],
        min_silence_ms: int,
        min_chunk_ms: int = 0,
        max_chunk_ms: int = 0,
    ) -> List[Dict]:
        """Creates pacing chunks based on min/max duration and silence gaps.

        This function builds chunks of words that respect the desired duration
        constraints, splitting on silences when possible.

        Returns a list of dicts with keys: 'raw_text', 'duration_ms', 'start_ms', 'end_ms'.
        """
        if not words:
            return []

        # Ensure min/max are valid to prevent infinite loops or no-ops
        if not min_chunk_ms or min_chunk_ms <= 0:
            min_chunk_ms = 5 * 1000  # Default to 5s
        if not max_chunk_ms or max_chunk_ms <= min_chunk_ms:
            max_chunk_ms = min_chunk_ms * 2  # Default to 2x min

        chunks = []
        current_chunk_start_index = 0

        # Iterate through words to find split points
        for i in range(len(words)):
            start_time = words[current_chunk_start_index]['start']
            end_time = words[i]['end']
            duration = (end_time - start_time) * 1000

            is_last_word = i == len(words) - 1
            
            # Check for a significant pause after the current word
            silence_after = (words[i + 1]['start'] - end_time) * 1000 if not is_last_word else 0

            # Determine if we should end the chunk at the current word
            split_here = False
            if is_last_word:
                split_here = True
            elif duration > max_chunk_ms:
                split_here = True
            elif duration >= min_chunk_ms and silence_after >= min_silence_ms:
                split_here = True

            if split_here:
                chunk_words = words[current_chunk_start_index : i + 1]
                text = " ".join([w['text'].strip() for w in chunk_words])
                
                start_ms = chunk_words[0]['start'] * 1000
                end_ms = chunk_words[-1]['end'] * 1000
                
                chunks.append({
                    'raw_text': text,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'duration_ms': end_ms - start_ms,
                })
                
                # The next chunk starts at the next word
                current_chunk_start_index = i + 1

        # If the last chunk is too short (because of a forced split by max_duration),
        # merge it with the previous one.
        if len(chunks) > 1 and chunks[-1]['duration_ms'] < min_chunk_ms:
            last_chunk = chunks.pop()
            prev_chunk = chunks[-1]
            
            prev_chunk['raw_text'] += " " + last_chunk['raw_text']
            prev_chunk['end_ms'] = last_chunk['end_ms']
            prev_chunk['duration_ms'] = prev_chunk['end_ms'] - prev_chunk['start_ms']

        return chunks