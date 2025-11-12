"""
Orchestrator Module

This file contains the Orchestrator class, which is the central coordinator for
the entire video generation pipeline.

Responsibilities:
- Load user input and technical configuration data.
- Initialize all the necessary processor tools.
- Manage the step-by-step workflow of the pipeline from script to video.
- Handle file I/O, saving intermediate assets and passing their paths
  between processors.
- Perform cleanup of temporary files after a run.
"""

import os
import json
import time
import shutil
import traceback
import re
import io
from pydub import AudioSegment

# Import all the processor tools
from processors.script_generator import ScriptGenerator
from processors.tts_processor import TTSProcessor
from processors.audio_analyzer import AudioAnalyzer
from processors.prompt_processor import PromptProcessor
from processors.image_generator import ImageGenerator
from processors.subtitle_processor import SubtitleProcessor
from processors.video_renderer import VideoRenderer


class Orchestrator:
    """
    Manages the entire video creation workflow from start to finish.
    """

    def __init__(self, config_path: str, input_path: str):
        """
        Initializes the Orchestrator.

        Args:
            config_path (str): Path to the main config.json file.
            input_path (str): Path to the consolidated input.json file.
            api_key (str): The Gemini API key (validated and passed to specific processors).
        """


        self.config = self._load_json(config_path)
        self.input_data = self._load_json(input_path)

        self.timestamp = str(int(time.time()))
        self._setup_directories()

        self.asset_paths = {}
        self.script_content = None

        # --- CORRECTED INITIALIZATIONS ---
        # Most processors are standardized to find the API key from the environment.
        # The ImageGenerator is treated as a special case, as requested, and is passed the key directly.
        print("Initializing processor tools...")
        self.script_generator = ScriptGenerator(self.config['gemini_models']['llm'])
        self.tts_processor = TTSProcessor(self.config['gemini_models']['tts'])
        self.audio_analyzer = AudioAnalyzer(model_size=self.config['audio_analysis']['stt_whisper_model_size'])
        self.prompt_processor = PromptProcessor(self.config['gemini_models']['llm'])
        
        # CORRECTED LINE: Pass both the api_key and model_name to the ImageGenerator.
        self.image_generator = ImageGenerator(
            model_name=self.config['gemini_models']['image_generator'],
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        self.subtitle_processor = SubtitleProcessor()
        self.video_renderer = VideoRenderer()
        print("--- Orchestrator Initialized Successfully ---")

    def _load_json(self, file_path: str) -> dict:
        """Loads a JSON file and handles potential errors."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _setup_directories(self):
        """Creates the necessary output directories for the run."""
        self.run_dir = os.path.join(self.config['directories']['output'], self.timestamp)
        self.image_dir = os.path.join(self.run_dir, "images")
        self.audio_dir = os.path.join(self.run_dir, "audio")
        self.temp_dir = os.path.join(self.run_dir, "temp")

        for dir_path in [self.run_dir, self.image_dir, self.audio_dir, self.temp_dir]:
            os.makedirs(dir_path, exist_ok=True)
        print(f"Created run directory: {self.run_dir}")

    def run_pipeline(self):
        """Executes the full video generation pipeline step-by-step."""
        print("\n--- Starting Fully Automated Video Generation Pipeline ---")
        start_time = time.time()

        try:
            self._generate_script()
            self._generate_audio()
            analysis_result = self._analyze_audio()
            self._generate_visuals(analysis_result['pacing_chunks'])
            self._generate_subtitles(analysis_result['word_timestamps'])
            self._render_video()

            end_time = time.time()
            print("\n--- Pipeline Finished Successfully! ---")
            print(f"Total execution time: {end_time - start_time:.2f} seconds.")
            print(f"Final video saved to: {self.asset_paths['final_video']}")
            return True

        except Exception as e:
            end_time = time.time()
            print(f"\n--- PIPELINE FAILED after {end_time - start_time:.2f} seconds. ---")
            print(f"An error occurred: {e}")
            traceback.print_exc()
            return False
        finally:
            if self.config.get('cleanup_temp_dir', True) and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                    print("Cleaned up temporary directory.")
                except OSError as e:
                    print(f"Error cleaning up temporary directory: {e}")

    def _generate_script(self):
        """Step 0: Script Generation"""
        print("\n[Step 0/6] Generating Video Script...")
        script_config = self.input_data['script_generation']
        self.script_content = self.script_generator.process(
            topics=script_config['topics'],
            keywords=script_config['keywords'],
            tone=script_config['tone'],
            target_word_count=script_config['target_word_count']
        )
        if not self.script_content:
            raise RuntimeError("Script generation failed. Aborting pipeline.")
        script_path = os.path.join(self.run_dir, "generated_script.txt")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(self.script_content)
        self.asset_paths['script'] = script_path
        print(f"  - Script saved for review at: {script_path}")

    def _generate_audio(self):
        """Step 1: Text-to-Speech"""
        print("\n[Step 1/6] Generating Master Audio...")
        audio_bytes = self.tts_processor.process(
            self.script_content, self.config['tts_settings']['voice_name']
        )
        if not audio_bytes:
            raise RuntimeError("TTS generation failed. Received no audio data.")
        
        audio_stream = io.BytesIO(audio_bytes)
        try:
            audio_segment = AudioSegment.from_file(audio_stream)
        except Exception as e:
            raise RuntimeError(f"Audio decoding failed. The TTS output may be invalid. Error: {e}")

        master_audio_path = os.path.join(self.audio_dir, "master_audio.wav")
        audio_segment.export(master_audio_path, format="wav")
        self.asset_paths['master_audio'] = master_audio_path
        print(f"  - Master audio file successfully converted and saved to: {master_audio_path}")

    def _analyze_audio(self):
        """Step 2: Audio Analysis"""
        print("\n[Step 2/6] Analyzing Audio for Pacing and Timestamps...")
        with open(self.asset_paths['master_audio'], 'rb') as f:
            wav_audio_bytes = f.read()
        analysis_result = self.audio_analyzer.process(
            wav_audio_bytes, self.config['audio_analysis']
        )
        return analysis_result

    def _generate_visuals(self, pacing_chunks):
        """Step 3: Visual Content Generation"""
        print("\n[Step 3/6] Generating Visuals (Prompts and Images)...")
        image_sequence = []
        for i, chunk in enumerate(pacing_chunks):
            print(f"  - Processing visual for chunk {i+1}/{len(pacing_chunks)}...")
            prompt = self.prompt_processor.process(
                text_chunk=chunk['raw_text'],
                creative_brief=self.input_data['style_brief']['creative_brief'],
                global_summary=self.script_content
            )
            image_bytes = self.image_generator.process(
                prompt, self.config['image_generation']['negative_prompt_terms']
            )
            if image_bytes:
                img_path = os.path.join(self.image_dir, f"image_{i:03d}.png")
                with open(img_path, 'wb') as f:
                    f.write(image_bytes)
                image_sequence.append({'path': img_path, 'duration_s': chunk['duration_ms'] / 1000.0})
        self.asset_paths['image_sequence'] = image_sequence

    def _generate_subtitles(self, word_timestamps):
        """Step 4: Subtitle Generation"""
        print("\n[Step 4/6] Generating Subtitles...")
        heading_strings = re.findall(r":(.*?)::", self.script_content)
        ass_content = self.subtitle_processor.process(
            word_timestamps=word_timestamps,
            heading_strings=heading_strings,
            style_config=self.config['subtitle_style'],
            heading_style_config=self.config.get('heading_style', {}),
            video_width=self.config['video_rendering']['target_width'],
            video_height=self.config['video_rendering']['target_height']
        )
        subtitle_path = os.path.join(self.run_dir, "subtitles.ass")
        with open(subtitle_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        self.asset_paths['subtitles'] = subtitle_path

    def _render_video(self):
        """Step 5: Final Video Rendering"""
        print("\n[Step 5/6] Assembling Final Video...")
        temp_video_path = os.path.join(self.temp_dir, "video_with_audio.mp4")
        final_output_filename = self.input_data['video_details'].get(
            'output_filename', f'final_video_{self.timestamp}.mp4'
        )
        final_output_path = os.path.join(self.run_dir, final_output_filename)

        success_assembly = self.video_renderer.assemble_primary_video(
            image_sequence=self.asset_paths['image_sequence'],
            audio_path=self.asset_paths['master_audio'],
            render_config=self.config['video_rendering'],
            output_path=temp_video_path
        )
        if not success_assembly:
            raise RuntimeError("Primary video assembly failed. Aborting.")

        success_burn = self.video_renderer.burn_subtitles(
            video_path=temp_video_path,
            subtitle_path=self.asset_paths['subtitles'],
            output_path=final_output_path
        )
        if not success_burn:
            raise RuntimeError("Subtitle burn-in failed.")

        self.asset_paths['final_video'] = final_output_path