"""
Subtitle Processor Module (Corrected with Original Notebook Logic)

This file contains the SubtitleProcessor class, a tool for generating a styled
subtitle file in the Advanced SubStation Alpha (.ass) format.

This version faithfully reimplements the tested and complex logic from the
original notebook (Cell 10), including fuzzy heading alignment and detailed
line-breaking rules, while conforming to our new modular architecture.
"""
import os
import re
import difflib
import traceback
from typing import List, Dict, Any

# We import the AudioAnalyzer here *only* for the fallback mechanism.
from .audio_analyzer import AudioAnalyzer

class SubtitleProcessor:
    """
    Generates a complete .ass subtitle file using the proven logic from the original project.
    """
    def _format_ass_time(self, seconds: float) -> str:
        """Helper to convert seconds into H:MM:SS.cs format for .ass files."""
        if not isinstance(seconds, (int, float)) or seconds < 0:
            seconds = 0.0
        milliseconds = round(seconds * 1000.0)
        hours = int(milliseconds // 3_600_000); milliseconds %= 3_600_000
        minutes = int(milliseconds // 60_000); milliseconds %= 60_000
        secs = int(milliseconds // 1_000); milliseconds %= 1_000
        centiseconds = int(milliseconds // 10)
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def _normalize_text_for_matching(self, text: str) -> str:
        """Helper for fuzzy matching, identical to notebook version."""
        if not text: return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _generate_header(self, style_config: Dict, heading_style_config: Dict, video_width: int, video_height: int) -> str:
        """Constructs the .ass header, now including a dedicated Heading style."""
        default_style = style_config.get('default', {})
        heading_style = heading_style_config.get('heading', {})
        
        default_style_line = (
            f"Style: Default,"
            f"{default_style.get('font_name', 'Arial')},{default_style.get('font_size', 72)},"
            f"{default_style.get('primary_colour', '&H00FFFFFF')},&H000000FF,"
            f"{default_style.get('outline_colour', '&H00000000')},{default_style.get('back_colour', '&H99000000')},"
            f"-1,0,0,0,100,100,0,0,1,2,1,"
            f"{default_style.get('alignment', 2)},10,10,20,1"
        )
        
        heading_style_line = (
            f"Style: HeadingStyle,"
            f"{heading_style.get('font_name', 'Arial')},{heading_style.get('font_size', 86)},"
            f"{heading_style.get('primary_colour', '&H00FFFF00')},&H000000FF,"
            f"{heading_style.get('outline_colour', '&H00000000')},{heading_style.get('back_colour', '&H60000000')},"
            f"-1,0,0,0,100,100,0,0,1,2,1,"
            f"{heading_style.get('alignment', 5)},30,30,50,1"
        )

        header = f"""[Script Info]
Title: Auto-Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{default_style_line}
{heading_style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        return header

    def _align_headings(self, heading_strings: List[str], word_timestamps: List[Dict]) -> (List[Dict], set):
        """Replicates the fuzzy matching logic from the notebook to find headings."""
        if not heading_strings or not word_timestamps:
            return [], set()

        print("Aligning identified headings with word timestamps...")
        stt_normalized_words = [self._normalize_text_for_matching(w['text']) for w in word_timestamps]
        aligned_headings = []
        used_word_indices = set()

        for heading_text in heading_strings:
            normalized_heading = self._normalize_text_for_matching(heading_text)
            if not normalized_heading: continue
            
            heading_words = normalized_heading.split()
            matcher = difflib.SequenceMatcher(None, heading_words, stt_normalized_words, autojunk=False)
            best_match = None
            highest_ratio = 0.7 # Minimum similarity threshold

            for match in matcher.get_matching_blocks():
                if match.size == 0: continue
                
                start_idx, end_idx = match.b, match.b + match.size - 1
                current_indices = set(range(start_idx, end_idx + 1))
                
                # Check if any words in this potential match are already used by a better heading match
                if current_indices.intersection(used_word_indices):
                    continue

                # Simple ratio, can be improved, but matches notebook's intent
                ratio = match.size / len(heading_words)
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = {
                        "text": heading_text, # Use original, un-normalized text
                        "start_s": word_timestamps[start_idx]['start'],
                        "end_s": word_timestamps[end_idx]['end'],
                        "indices": current_indices,
                        "ratio": ratio
                    }
            
            if best_match:
                # Check for overlap again before committing
                if not best_match["indices"].intersection(used_word_indices):
                    aligned_headings.append(best_match)
                    used_word_indices.update(best_match["indices"])
                    print(f"  - Aligned heading (ratio {best_match['ratio']:.2f}): \"{best_match['text'][:50]}...\"")

        aligned_headings.sort(key=lambda x: x['start_s'])
        print(f"Alignment complete. {len(aligned_headings)} headings successfully aligned.")
        return aligned_headings, used_word_indices


    def process(self, **kwargs) -> str:
        """
        Generates the full content of an .ass subtitle file using the original notebook's logic.

        Args (via kwargs):
            style_config (Dict): 'subtitle_style' section from config.json.
            heading_style_config (Dict): 'heading_style' section from config.json.
            video_width (int), video_height (int): Target video dimensions.
            word_timestamps (List[Dict]): Pre-processed word timestamp data.
            heading_strings (List[str], optional): List of heading texts to align.
            time_offset_s (float, optional): Global time offset for all subtitles (e.g., for intro).

            --- Fallback Parameters ---
            audio_bytes, analyzer_config, analyzer_model_size
        """
        print("Generating subtitles using notebook-derived logic...")
        word_timestamps = kwargs.get('word_timestamps')

        # --- Fallback Mechanism ---
        if not word_timestamps:
            print("INFO (SubtitleProcessor): No pre-processed timestamp data. Engaging fallback audio analysis.")
            if not kwargs.get('audio_bytes') or not kwargs.get('analyzer_config'):
                raise ValueError("Fallback requires 'audio_bytes' and 'analyzer_config'.")
            
            try:
                analyzer = AudioAnalyzer(model_size=kwargs.get('analyzer_model_size', 'small'))
                analysis_result = analyzer.process(kwargs['audio_bytes'], kwargs['analyzer_config'])
                word_timestamps = analysis_result.get('word_timestamps')
                if not word_timestamps:
                    print("Warning (SubtitleProcessor): Fallback analysis yielded no timestamps. Returning empty file.")
                    return self._generate_header(kwargs['style_config'], kwargs.get('heading_style_config',{}), kwargs['video_width'], kwargs['video_height'])
            except Exception as e:
                raise RuntimeError("Fallback audio analysis failed.") from e

        # --- Main Logic (Replicated from Notebook) ---
        style_config = kwargs.get('style_config', {})
        heading_style_config = kwargs.get('heading_style_config', {})
        video_width = kwargs.get('video_width', 1920)
        video_height = kwargs.get('video_height', 1080)
        heading_strings = kwargs.get('heading_strings', [])
        time_offset_s = kwargs.get('time_offset_s', 0.0)
        
        header = self._generate_header(style_config, heading_style_config, video_width, video_height)
        
        aligned_headings, used_word_indices = self._align_headings(heading_strings, word_timestamps)
        
        # 1. Add heading dialogue entries
        dialogue_entries = []
        for heading in aligned_headings:
            start_time = self._format_ass_time(heading['start_s'] + time_offset_s)
            end_time = self._format_ass_time(heading['end_s'] + time_offset_s)
            text = heading['text'].replace('\n', '\\N') # Allow newlines in headings
            entry = f"Dialogue: 0,{start_time},{end_time},HeadingStyle,,0,0,0,,{text}"
            dialogue_entries.append(entry)

        # 2. Group remaining words into lines and add as default entries
        default_line_rules = style_config.get('line_rules', {})
        max_words = default_line_rules.get('max_words_per_line', 7)
        max_duration = default_line_rules.get('max_line_duration_s', 12.0)
        gap_threshold = default_line_rules.get('gap_threshold_s', 0.4)
        
        current_line_words = []
        for i, word in enumerate(word_timestamps):
            if i in used_word_indices: # Skip words used in headings
                if current_line_words: # Finalize pending line before skipping
                    start_s = current_line_words[0]['start']
                    end_s = current_line_words[-1]['end']
                    text = " ".join(w['text'] for w in current_line_words)
                    entry = f"Dialogue: 0,{self._format_ass_time(start_s + time_offset_s)},{self._format_ass_time(end_s + time_offset_s)},Default,,0,0,0,,{text}"
                    dialogue_entries.append(entry)
                    current_line_words = []
                continue

            # Line breaking logic
            start_new_line = False
            if not current_line_words:
                start_new_line = True
            else:
                gap = word['start'] - word_timestamps[i-1]['end']
                line_dur = word['end'] - current_line_words[0]['start']
                if len(current_line_words) >= max_words or line_dur > max_duration or gap > gap_threshold:
                    start_new_line = True
            
            if start_new_line and current_line_words:
                start_s = current_line_words[0]['start']
                end_s = current_line_words[-1]['end']
                text = " ".join(w['text'] for w in current_line_words)
                entry = f"Dialogue: 0,{self._format_ass_time(start_s + time_offset_s)},{self._format_ass_time(end_s + time_offset_s)},Default,,0,0,0,,{text}"
                dialogue_entries.append(entry)
                current_line_words = []
            
            current_line_words.append(word)

        if current_line_words: # Add the very last line
            start_s = current_line_words[0]['start']
            end_s = current_line_words[-1]['end']
            text = " ".join(w['text'] for w in current_line_words)
            entry = f"Dialogue: 0,{self._format_ass_time(start_s + time_offset_s)},{self._format_ass_time(end_s + time_offset_s)},Default,,0,0,0,,{text}"
            dialogue_entries.append(entry)

        # Sort all entries by start time to ensure correct order
        dialogue_entries.sort(key=lambda x: x.split(',')[1])

        full_content = header + "\n".join(dialogue_entries)
        print(f"Subtitle generation complete. Created {len(dialogue_entries)} dialogue lines.")
        return full_content