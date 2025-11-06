"""
Orchestrator Module

This file contains the Orchestrator class, which is the central coordinator for
the entire video generation pipeline.

Responsibilities:
- Load the user input and configuration data.
- Initialize all the necessary processor tools.
- Manage the step-by-step workflow of the pipeline.
- Handle file I/O, saving intermediate assets and passing their paths
  between processors.
- Perform cleanup of temporary files.
"""
import os
import json
import time
import shutil

# Import all the processor tools we have built
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
    def __init__(self, config_path: str, user_input_path: str):
        """
        Initializes the Orchestrator by loading configuration and user input.

        Args:
            config_path (str): Path to the main config.json file.
            user_input_path (str): Path to the user_input.json file for a specific run.
        """
        print("--- Initializing Orchestrator ---")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not os.path.exists(user_input_path):
            raise FileNotFoundError(f"User input file not found: {user_input_path}")
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        with open(user_input_path, 'r') as f:
            self.user_input = json.load(f)

        self.timestamp = str(int(time.time()))
        self._setup_directories()
        self.asset_paths = {} # To store paths of all generated assets

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
        print("\n--- Starting Video Generation Pipeline ---")
        start_time = time.time()
        
        try:
            # --- Step 1: Text-to-Speech ---
            print("\n[Step 1/5] Generating Master Audio...")
            tts_tool = TTSProcessor(
                api_key=self.user_input['api_key'],
                model_name=self.config['gemini_models']['tts']
            )
            audio_bytes = tts_tool.process(
                text_to_speak=self.user_input['script_text'],
                voice_name=self.config['tts_settings']['voice_name']
            )
            master_audio_path = os.path.join(self.audio_dir, "master_audio.wav")
            with open(master_audio_path, 'wb') as f:
                f.write(audio_bytes)
            self.asset_paths['master_audio'] = master_audio_path

            # --- Step 2: Audio Analysis ---
            print("\n[Step 2/5] Analyzing Audio for Pacing and Timestamps...")
            analyzer_tool = AudioAnalyzer(model_size=self.config['audio_analysis']['stt_whisper_model_size'])
            analysis_result = analyzer_tool.process(
                audio_bytes,
                self.config['audio_analysis']
            )
            pacing_chunks = analysis_result['pacing_chunks']
            word_timestamps = analysis_result['word_timestamps']

            # --- Step 3: Visual Content Generation ---
            print("\n[Step 3/5] Generating Visuals (Prompts and Images)...")
            prompt_tool = PromptProcessor(
                api_key=self.user_input['api_key'],
                model_name=self.config['gemini_models']['llm']
            )
            image_tool = ImageGenerator(
                api_key=self.user_input['api_key'],
                model_name=self.config['gemini_models']['image_generator']
            )
            
            image_sequence = []
            for i, chunk in enumerate(pacing_chunks):
                print(f"  - Processing visual for chunk {i+1}/{len(pacing_chunks)}...")
                prompt = prompt_tool.process(
                    text_chunk=chunk['raw_text'],
                    creative_brief=self.user_input['creative_brief'],
                    global_summary=self.user_input['script_text'] # Using full script as summary for simplicity
                )
                image_bytes = image_tool.process(
                    prompt,
                    self.config['image_generation']['negative_prompt_terms']
                )
                if image_bytes:
                    img_path = os.path.join(self.image_dir, f"image_{i:03d}.png")
                    with open(img_path, 'wb') as f:
                        f.write(image_bytes)
                    image_sequence.append({'path': img_path, 'duration_s': chunk['duration_ms'] / 1000.0})
            self.asset_paths['image_sequence'] = image_sequence
            
            # --- Step 4: Subtitle Generation ---
            print("\n[Step 4/5] Generating Subtitles...")
            subtitle_tool = SubtitleProcessor()
            ass_content = subtitle_tool.process(
                style_config=self.config['subtitle_style'],
                video_width=self.config['video_rendering']['target_width'],
                video_height=self.config['video_rendering']['target_height'],
                word_timestamps=word_timestamps
            )
            subtitle_path = os.path.join(self.run_dir, "subtitles.ass")
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            self.asset_paths['subtitles'] = subtitle_path

            # --- Step 5: Final Video Rendering ---
            print("\n[Step 5/5] Assembling Final Video...")
            renderer_tool = VideoRenderer()
            temp_video_path = os.path.join(self.temp_dir, "video_with_audio.mp4")
            final_output_path = os.path.join(self.run_dir, self.user_input.get('output_filename', f'final_video_{self.timestamp}.mp4'))
            
            # First, assemble the video with audio
            success_assembly = renderer_tool.assemble_primary_video(
                image_sequence=self.asset_paths['image_sequence'],
                audio_path=self.asset_paths['master_audio'],
                render_config=self.config['video_rendering'],
                output_path=temp_video_path
            )
            
            if not success_assembly:
                raise RuntimeError("Primary video assembly failed. Aborting.")
            
            # Then, burn the subtitles
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

        except Exception as e:
            end_time = time.time()
            print(f"\n--- PIPELINE FAILED after {end_time - start_time:.2f} seconds. ---")
            print(f"An error occurred: {e}")
            traceback.print_exc()
        finally:
            # --- Cleanup ---
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print("Cleaned up temporary directory.")