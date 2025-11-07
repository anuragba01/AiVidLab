"""
Orchestrator Module

This file contains the Orchestrator class, which is the central coordinator for
the entire video generation pipeline.

Responsibilities:
- Load the user input and technical configuration data.
- Initialize all the necessary processor tools using a secure API key.
- Manage the step-by-step workflow of the pipeline from script to video.
- Handle all file I/O, saving intermediate assets and passing their paths
  between processors.
- Perform cleanup of temporary files after a run.
"""

import os
import json
import time
import shutil
import traceback
import re  # For extracting headings from the script
import io
from pydub import AudioSegment

# Import all the processor tools we have built
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

    def __init__(self, config_path: str, input_path: str, api_key: str):
        """
        Initializes the Orchestrator.

        Args:
            config_path (str): Path to the main config.json file.
            input_path (str): Path to the consolidated input.json file.
            api_key (str): The Gemini API key loaded from the environment.
        """
        print("--- Initializing Orchestrator ---")
        if not api_key:
            raise ValueError("API key was not provided. Please ensure it is in your .env file.")
        self.api_key = api_key

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)
        with open(input_path, 'r') as f:
            self.input_data = json.load(f)

        self.timestamp = str(int(time.time()))
        self._setup_directories()
        self.asset_paths = {}  # To store paths of all generated assets
        self.script_content = None  # To hold the generated script in memory

    def _setup_directories(self):
        """Creates the necessary output directories for the run."""
        self.run_dir = os.path.join(self.config['directories']['output'], self.timestamp)
        self.image_dir = os.path.join(self.run_dir, "images")
        self.audio_dir = os.path.join(self.run_dir, "audio")
        self.temp_dir = os.path.join(self.run_dir, "temp")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"Created run directory: {self.run_dir}")

    def run_pipeline(self):
        """
        Executes the full video generation pipeline step-by-step.
        """
        print("\n--- Starting Fully Automated Video Generation Pipeline ---")
        start_time = time.time()

        try:
            # --- Step 0: Script Generation ---
            print("\n[Step 0/6] Generating Video Script...")
            script_config = self.input_data['script_generation']
            script_tool = ScriptGenerator(self.api_key, self.config['gemini_models']['llm'])
            self.script_content = script_tool.process(
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

            # --- Step 1: Text-to-Speech ---
            print("\n[Step 1/6] Generating Master Audio...")
            tts_tool = TTSProcessor(self.api_key, self.config['gemini_models']['tts'])
            audio_bytes = tts_tool.process(
                self.script_content, self.config['tts_settings']['voice_name']
            )

            # Convert and save master audio as WAV
            audio_stream = io.BytesIO(audio_bytes)


            audio_stream = io.BytesIO(audio_bytes)

            try:
                # Try guessing the format
                audio_segment = AudioSegment.from_file(audio_stream, format="mp3")
            except Exception as e:
                print("Could not decode as MP3, trying WAV...")
                audio_stream.seek(0)
                try:
                    audio_segment = AudioSegment.from_file(audio_stream, format="wav")
                except Exception as e2:
                    print(" Failed to decode audio stream. Likely invalid data from TTS.")
                    raise RuntimeError(f"Audio decoding failed: {e2}")


            master_audio_path = os.path.join(self.audio_dir, "master_audio.wav")
            audio_segment.export(master_audio_path, format="wav")

            self.asset_paths['master_audio'] = master_audio_path
            print(f"  - Master audio file successfully converted and saved to: {master_audio_path}")

            # --- Step 2: Audio Analysis ---
            print("\n[Step 2/6] Analyzing Audio for Pacing and Timestamps...")
            analyzer_tool = AudioAnalyzer(model_size=self.config['audio_analysis']['stt_whisper_model_size'])

            with open(master_audio_path, 'rb') as f:
                wav_audio_bytes = f.read()

            analysis_result = analyzer_tool.process(
                wav_audio_bytes, self.config['audio_analysis']
            )

            pacing_chunks = analysis_result['pacing_chunks']
            word_timestamps = analysis_result['word_timestamps']

            # --- Step 3: Visual Content Generation ---
            print("\n[Step 3/6] Generating Visuals (Prompts and Images)...")
            prompt_tool = PromptProcessor(self.api_key, self.config['gemini_models']['llm'])
            image_tool = ImageGenerator(self.api_key, self.config['gemini_models']['image_generator'])

            image_sequence = []
            for i, chunk in enumerate(pacing_chunks):
                print(f"  - Processing visual for chunk {i+1}/{len(pacing_chunks)}...")
                prompt = prompt_tool.process(
                    text_chunk=chunk['raw_text'],
                    creative_brief=self.input_data['style_brief']['creative_brief'],
                    global_summary=self.script_content
                )
                image_bytes = image_tool.process(
                    prompt, self.config['image_generation']['negative_prompt_terms']
                )
                if image_bytes:
                    img_path = os.path.join(self.image_dir, f"image_{i:03d}.png")
                    with open(img_path, 'wb') as f:
                        f.write(image_bytes)
                    image_sequence.append({'path': img_path, 'duration_s': chunk['duration_ms'] / 1000.0})
            self.asset_paths['image_sequence'] = image_sequence

            # --- Step 4: Subtitle Generation ---
            print("\n[Step 4/6] Generating Subtitles...")
            heading_strings = re.findall(r":(.*?)::", self.script_content)

            subtitle_tool = SubtitleProcessor()
            ass_content = subtitle_tool.process(
                word_timestamps=word_timestamps,
                heading_strings=heading_strings,
                style_config={
                    'default': self.config['subtitle_style'],
                    'line_rules': self.config['subtitle_style']
                },
                heading_style_config={'heading': self.config.get('heading_style', {})},
                video_width=self.config['video_rendering']['target_width'],
                video_height=self.config['video_rendering']['target_height']
            )
            subtitle_path = os.path.join(self.run_dir, "subtitles.ass")
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            self.asset_paths['subtitles'] = subtitle_path

            # --- Step 5: Final Video Rendering ---
            print("\n[Step 5/6] Assembling Final Video...")
            renderer_tool = VideoRenderer()
            temp_video_path = os.path.join(self.temp_dir, "video_with_audio.mp4")

            final_output_filename = self.input_data['video_details'].get(
                'output_filename', f'final_video_{self.timestamp}.mp4'
            )
            final_output_path = os.path.join(self.run_dir, final_output_filename)

            success_assembly = renderer_tool.assemble_primary_video(
                image_sequence=self.asset_paths['image_sequence'],
                audio_path=self.asset_paths['master_audio'],
                render_config=self.config['video_rendering'],
                output_path=temp_video_path
            )

            if not success_assembly:
                raise RuntimeError("Primary video assembly failed. Aborting.")

            success_burn = renderer_tool.burn_subtitles(
                video_path=temp_video_path,
                subtitle_path=self.asset_paths['subtitles'],
                output_path=final_output_path
            )

            if not success_burn:
                raise RuntimeError("Subtitle burn-in failed.")

            self.asset_paths['final_video'] = final_output_path

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

        # finally:
        #     if os.path.exists(self.temp_dir):
        #         try:
        #             shutil.rmtree(self.temp_dir)
        #             print("Cleaned up temporary directory.")
        #         except OSError as e:
        #             print(f"Error cleaning up temporary directory: {e}")
