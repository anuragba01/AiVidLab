"""
Audio Analyzer Processor Module

Processes raw audio into transcriptions and pacing data for downstream video tasks.
Outputs:
1. Pacing chunks (timing for scenes)
2. Word-level timestamps (for subtitles)
"""

import os
import io
import tempfile
import traceback
from typing import List, Dict, Any

try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
except ImportError:
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
            audio_stream = io.BytesIO(audio_bytes)
            audio_segment = AudioSegment.from_file(audio_stream)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                audio_segment.export(tmpfile.name, format="wav")
                tmpfile.flush()

                audio_for_ts = whisper.load_audio(tmpfile.name)
                result = whisper.transcribe(self.model, audio_for_ts, language="en")

                word_timestamps = self._extract_word_timestamps(result)
                audio_segment = AudioSegment.from_file(tmpfile.name, format="wav")
                silence_chunks = self._chunk_on_silence(audio_segment, chunk_config)
                pacing_chunks = self._align_text_to_chunks(silence_chunks, word_timestamps)

                return {
                    'pacing_chunks': pacing_chunks,
                    'word_timestamps': word_timestamps
                }

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Audio analysis failed: {e}") from e

    def _extract_word_timestamps(self, transcription_result: Dict) -> List[Dict[str, Any]]:
        """Flattens Whisper result into a list of words."""
        return [
            {'text': w['text'], 'start': w['start'], 'end': w['end']}
            for seg in transcription_result.get("segments", [])
            for w in seg.get("words", [])
        ]

    def _chunk_on_silence(self, audio_segment: AudioSegment, config: Dict) -> List[AudioSegment]:
        """Splits audio based on silence using Pydub."""
        silence_thresh = audio_segment.dBFS + config.get('silence_thresh_offset', -16)
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=config.get('min_silence_len_ms', 500),
            silence_thresh=silence_thresh,
            keep_silence=250
        )
        return chunks or [audio_segment]

    def _align_text_to_chunks(self, audio_chunks: List[AudioSegment], words: List[Dict]) -> List[Dict]:
        """Maps transcribed words to their respective audio chunks."""
        pacing_chunks = []
        word_idx = 0
        current_time_ms = 0

        for chunk in audio_chunks:
            chunk_duration_ms = len(chunk)
            chunk_end_time_s = (current_time_ms + chunk_duration_ms) / 1000.0
            chunk_text_parts = []

            while word_idx < len(words) and words[word_idx]['start'] < chunk_end_time_s:
                chunk_text_parts.append(words[word_idx]['text'])
                word_idx += 1

            if chunk_text_parts:
                pacing_chunks.append({
                    'raw_text': " ".join(chunk_text_parts),
                    'duration_ms': chunk_duration_ms
                })

            current_time_ms += chunk_duration_ms

        return pacing_chunks


if __name__ == '__main__':
    test_audio_file = "out.wav"

    if not os.path.exists(test_audio_file):
        print(f"Missing test audio: {test_audio_file}")
    else:
        TEST_MODEL_SIZE = "tiny"
        TEST_CHUNK_CONFIG = {"min_silence_len_ms": 400, "silence_thresh_offset": -16}

        with open(test_audio_file, "rb") as f:
            audio_data = f.read()

        analyzer = AudioAnalyzer(model_size=TEST_MODEL_SIZE)
        result = analyzer.process(audio_data, TEST_CHUNK_CONFIG)

        if result['word_timestamps'] and result['pacing_chunks']:
            print("Analysis complete:")
            print(f"- {len(result['pacing_chunks'])} pacing chunks")
            print(f"- {len(result['word_timestamps'])} words")
        else:
            print("Analysis failed: empty result.")
