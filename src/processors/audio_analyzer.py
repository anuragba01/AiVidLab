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
        """Creates pacing chunks by finding large gaps between timed words.

        After initial chunking by silence gaps, the function will:
        - Merge adjacent chunks that are shorter than min_chunk_ms.
        - Split chunks longer than max_chunk_ms into multiple chunks on word boundaries.

        Returns a list of dicts with keys: 'raw_text', 'duration_ms', 'start_ms', 'end_ms'.
        """
        if not words:
            return []
        pacing_chunks = []
        current_chunk_word_indices = []

        for i, word in enumerate(words):
            current_chunk_word_indices.append(i)

            # Check for a long silence *after* the current word
            is_last_word = (i == len(words) - 1)
            if not is_last_word:
                next_word = words[i + 1]
                silence_duration = (next_word['start'] - word['end']) * 1000
                if silence_duration >= min_silence_ms:
                    # End of a chunk, save indices
                    start_idx = current_chunk_word_indices[0]
                    end_idx = current_chunk_word_indices[-1]
                    chunk_start_time = words[start_idx]['start']
                    chunk_end_time = words[end_idx]['end']
                    chunk_text = " ".join([words[j]['text'].strip() for j in current_chunk_word_indices])
                    chunk_duration_ms = (chunk_end_time - chunk_start_time) * 1000

                    pacing_chunks.append({
                        'raw_text': chunk_text,
                        'duration_ms': chunk_duration_ms,
                        'start_ms': chunk_start_time * 1000,
                        'end_ms': chunk_end_time * 1000,
                        'word_indices': (start_idx, end_idx)
                    })
                    current_chunk_word_indices = []  # Reset for next chunk
            else:
                # Last word ends the final chunk
                start_idx = current_chunk_word_indices[0]
                end_idx = current_chunk_word_indices[-1]
                chunk_start_time = words[start_idx]['start']
                chunk_end_time = words[end_idx]['end']
                chunk_text = " ".join([words[j]['text'].strip() for j in current_chunk_word_indices])
                chunk_duration_ms = (chunk_end_time - chunk_start_time) * 1000

                pacing_chunks.append({
                    'raw_text': chunk_text,
                    'duration_ms': chunk_duration_ms,
                    'start_ms': chunk_start_time * 1000,
                    'end_ms': chunk_end_time * 1000,
                    'word_indices': (start_idx, end_idx)
                })

        # Post-process: merge chunks shorter than min_chunk_ms
        if min_chunk_ms and pacing_chunks:
            merged = []
            i = 0
            while i < len(pacing_chunks):
                current = pacing_chunks[i]
                # If current chunk is long enough, keep it
                if current['duration_ms'] >= min_chunk_ms or i == len(pacing_chunks) - 1:
                    merged.append(current)
                    i += 1
                    continue

                # Otherwise, merge with the next chunk(s) until duration >= min_chunk_ms
                j = i + 1
                merged_chunk = dict(current)
                while j < len(pacing_chunks) and merged_chunk['duration_ms'] < min_chunk_ms:
                    next_chunk = pacing_chunks[j]
                    # merge text and extend end_ms
                    merged_chunk['raw_text'] = merged_chunk['raw_text'] + ' ' + next_chunk['raw_text']
                    merged_chunk['end_ms'] = next_chunk['end_ms']
                    merged_chunk['duration_ms'] = merged_chunk['end_ms'] - merged_chunk['start_ms']
                    merged_chunk['word_indices'] = (merged_chunk['word_indices'][0], next_chunk['word_indices'][1])
                    j += 1

                merged.append(merged_chunk)
                i = j

            pacing_chunks = merged

        # Post-process: split chunks longer than max_chunk_ms into multiple chunks
        if max_chunk_ms and pacing_chunks:
            split_chunks = []
            for chunk in pacing_chunks:
                if chunk['duration_ms'] <= max_chunk_ms:
                    split_chunks.append(chunk)
                    continue

                # Need to split by word boundaries between start and end indices
                start_idx, end_idx = chunk['word_indices']
                word_block = words[start_idx:end_idx + 1]

                # Build sub-chunks by accumulating words until they reach max_chunk_ms
                sub_start_idx = start_idx
                for k in range(len(word_block)):
                    candidate_end_idx = start_idx + k
                    sub_start_time = words[sub_start_idx]['start']
                    sub_end_time = words[candidate_end_idx]['end']
                    sub_duration = (sub_end_time - sub_start_time) * 1000
                    if sub_duration >= max_chunk_ms:
                        # create sub-chunk up to candidate_end_idx
                        sub_text = ' '.join([w['text'].strip() for w in words[sub_start_idx:candidate_end_idx + 1]])
                        split_chunks.append({
                            'raw_text': sub_text,
                            'duration_ms': sub_duration,
                            'start_ms': sub_start_time * 1000,
                            'end_ms': sub_end_time * 1000,
                            'word_indices': (sub_start_idx, candidate_end_idx)
                        })
                        sub_start_idx = candidate_end_idx + 1

                # Any remainder
                if sub_start_idx <= end_idx:
                    sub_start_time = words[sub_start_idx]['start']
                    sub_end_time = words[end_idx]['end']
                    sub_text = ' '.join([w['text'].strip() for w in words[sub_start_idx:end_idx + 1]])
                    split_chunks.append({
                        'raw_text': sub_text,
                        'duration_ms': (sub_end_time - sub_start_time) * 1000,
                        'start_ms': sub_start_time * 1000,
                        'end_ms': sub_end_time * 1000,
                        'word_indices': (sub_start_idx, end_idx)
                    })

            pacing_chunks = split_chunks

        # Finalize format: remove helper fields and ensure only requested keys returned
        final_chunks = []
        for c in pacing_chunks:
            final_chunks.append({
                'raw_text': c['raw_text'],
                'duration_ms': float(c['duration_ms'])
            })

        return final_chunks