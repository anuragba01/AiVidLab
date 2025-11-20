"""
Orchestrator Module - Production Ready

Manages video generation pipeline with proper error handling, logging,
and state management without overcomplicating.
"""

import os
import json
import time
import shutil
import traceback
import re
import logging
from enum import Enum
from typing import Dict, List, Any

from .processors.script_generator import ScriptGenerator
from .processors.tts_processor import TTSProcessor
from .processors.audio_analyzer import AudioAnalyzer
from .processors.prompt_processor import PromptProcessor
from .processors.image_generator import ImageGenerator
from .processors.subtitle_processor import SubtitleProcessor
from .processors.video_renderer import VideoRenderer
from .utilities.fallback_image_generator import generate_image_with_bytez

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages for tracking and recovery."""
    INIT = "initialization"
    SCRIPT = "script_generation"
    AUDIO = "audio_generation"
    ANALYSIS = "audio_analysis"
    PROMPTS = "prompt_generation"
    VISUALS = "visual_generation"
    SUBTITLES = "subtitle_generation"
    RENDER = "video_rendering"
    COMPLETE = "complete"


class Orchestrator:
    """Manages video creation workflow with proper error handling and state tracking."""

    def __init__(self, config_path: str, input_path: str):
        self.config = self._load_json(config_path)
        self.input_data = self._load_json(input_path)
        self._setup_directories()
        self._initialize_processors()
        
        self.asset_paths = {}
        self.script_content = None
        self.current_stage = PipelineStage.INIT
        
        logger.info("Orchestrator initialized successfully")

    def _load_json(self, file_path: str) -> dict:
        """Loads JSON with validation."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")

    def _setup_directories(self):
        """Creates output directory structure."""
        self.run_dir = os.path.abspath(self.config['directories']['output'])
        self.image_dir = os.path.join(self.run_dir, "images")
        self.audio_dir = os.path.join(self.run_dir, "audio")
        self.temp_dir = os.path.join(self.run_dir, "temp")
        for d in (self.image_dir, self.audio_dir, self.temp_dir):
            os.makedirs(d, exist_ok=True)

    def _initialize_processors(self):
        """Initializes all processor modules."""
        try:
            self.script_generator = ScriptGenerator(self.config['gemini_models']['m_llm'])
            self.tts_processor = TTSProcessor(self.config['gemini_models']['tts'])
            self.audio_analyzer = AudioAnalyzer(
                model_size=self.config['audio_analysis']['stt_whisper_model_size']
            )
            self.prompt_processor = PromptProcessor(self.config['gemini_models']['llm'])
            self.image_generator = ImageGenerator(
                model_name=self.config['gemini_models']['image_generator'],
                api_key=os.getenv("GEMINI_API_KEY")
            )
            self.subtitle_processor = SubtitleProcessor()
            self.video_renderer = VideoRenderer()
        except Exception as e:
            logger.error(f"Processor initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize processors: {e}")

    def _save_file(self, path: str, content: Any, mode: str = 'w'):
        """Centralized file saving with error handling."""
        try:
            with open(path, mode, encoding='utf-8' if 'b' not in mode else None) as f:
                f.write(content)
            logger.debug(f"Saved file: {path}")
        except IOError as e:
            logger.error(f"Failed to save {path}: {e}")
            raise

    def _save_state(self):
        """Saves pipeline state for potential recovery."""
        state = {
            'current_stage': self.current_stage.value,
            'asset_paths': self.asset_paths
        }
        state_path = os.path.join(self.run_dir, 'pipeline_state.json')
        self._save_file(state_path, json.dumps(state, indent=2))

    def run_pipeline(self) -> bool:
        """Executes full pipeline with error tracking."""
        logger.info("Starting video generation pipeline")
        start_time = time.time()
        
        pipeline_steps = [
            (PipelineStage.SCRIPT, self._generate_script),
            (PipelineStage.AUDIO, self._generate_audio),
            (PipelineStage.ANALYSIS, self._analyze_audio),
            (PipelineStage.PROMPTS, self._generate_prompts),
            (PipelineStage.VISUALS, self._generate_visuals),
            (PipelineStage.SUBTITLES, self._generate_subtitles),
            (PipelineStage.RENDER, self._render_video),
        ]

        try:
            analysis_result = None
            prompts_result = None
            
            for stage, step_func in pipeline_steps:
                self.current_stage = stage
                logger.info(f"Executing stage: {stage.value}")

                if stage == PipelineStage.SCRIPT:
                    script_path = os.path.join(self.run_dir, "generated_script.txt")
                    if os.path.exists(script_path):
                        logger.info("Script already exists, skipping script generation")
                        with open(script_path, 'r', encoding='utf-8') as f:
                            self.script_content = f.read()
                        self.asset_paths['script'] = script_path
                    else:
                        step_func()

                elif stage == PipelineStage.AUDIO:
                    master_audio_path = os.path.join(self.audio_dir, "master_audio.wav")
                    if os.path.exists(master_audio_path):
                        logger.info("Audio already exists, skipping TTS generation")
                        self.asset_paths['master_audio'] = master_audio_path
                    else:
                        step_func()

                elif stage == PipelineStage.ANALYSIS:
                    analysis_path = os.path.join(self.run_dir, 'analysis.json')
                    if os.path.exists(analysis_path):
                        logger.info("Analysis already exists, loading cached analysis")
                        with open(analysis_path, 'r', encoding='utf-8') as f:
                            analysis_result = json.load(f)
                    else:
                        analysis_result = step_func()

                elif stage == PipelineStage.PROMPTS:
                    prompts_path = os.path.join(self.run_dir, 'prompts.json')
                    if os.path.exists(prompts_path):
                        logger.info("Prompts already exist, loading cached prompts")
                        with open(prompts_path, 'r', encoding='utf-8') as f:
                            prompts_result = json.load(f)
                        self.asset_paths['prompts'] = prompts_path
                    else:
                        prompts_result = step_func(analysis_result['pacing_chunks'])

                elif stage == PipelineStage.VISUALS:
                    image_seq_path = os.path.join(self.run_dir, 'image_sequence.json')
                    if os.path.exists(image_seq_path):
                        logger.info("Image sequence already exists, skipping visual generation")
                        with open(image_seq_path, 'r', encoding='utf-8') as f:
                            self.asset_paths['image_sequence'] = json.load(f)
                    else:
                        step_func(prompts_result)

                elif stage == PipelineStage.SUBTITLES:
                    subtitle_path = os.path.join(self.run_dir, 'subtitles.ass')
                    if os.path.exists(subtitle_path):
                        logger.info("Subtitles already exist, skipping subtitle generation")
                        self.asset_paths['subtitles'] = subtitle_path
                    else:
                        step_func(analysis_result['word_timestamps'])

                elif stage == PipelineStage.RENDER:
                    final_output_filename = self.input_data['video_details'].get('output_filename', 'final_video.mp4')
                    final_output_path = os.path.join(self.run_dir, final_output_filename)
                    if os.path.exists(final_output_path):
                        logger.info("Final video already exists, skipping rendering")
                        self.asset_paths['final_video'] = final_output_path
                    else:
                        step_func()
                
                self._save_state()

            self.current_stage = PipelineStage.COMPLETE
            duration = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {duration:.2f}s")
            logger.info(f"Final video: {self.asset_paths.get('final_video', 'UNKNOWN')}")
            
            try:
                self._cleanup()
            except Exception:
                logger.exception("Cleanup failed after successful run (non-critical)")

            return True

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Pipeline failed at {self.current_stage.value} after {duration:.2f}s")
            logger.error(f"Error: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _generate_script(self):
        """Step 0: Script Generation"""
        logger.info("Generating script...")
        config = self.input_data['script_generation']
        self.script_content = self.script_generator.process(
            topics=config['topics'],
            keywords=config['keywords'],
            tone=config['tone'],
            target_word_count=config['target_word_count']
        )
        if not self.script_content or not self.script_content.strip():
            raise RuntimeError("Script generation returned empty content")
        script_path = os.path.join(self.run_dir, "generated_script.txt")
        self._save_file(script_path, self.script_content)
        self.asset_paths['script'] = script_path

    def _generate_audio(self):
        """Step 1: Text-to-Speech"""
        logger.info("Generating audio...")
        audio_bytes = self.tts_processor.process(
            self.script_content,
            self.config['tts_settings']['voice_name']
        )
        if not audio_bytes:
            raise RuntimeError("TTS returned no audio data")
        master_audio_path = os.path.join(self.audio_dir, "master_audio.wav")
        self.tts_processor.wave_file(master_audio_path, audio_bytes)
        self.asset_paths['master_audio'] = master_audio_path
        logger.info(f"Audio saved: {master_audio_path}")

    def _analyze_audio(self):
        """Step 2: Audio analysis"""
        logger.info("Analyzing audio...")
        master_audio_path = self.asset_paths['master_audio']
        chunk_config = self.config.get('audio_analysis', {})
        with open(master_audio_path, 'rb') as f:
            audio_bytes = f.read()
        analysis_result = self.audio_analyzer.process(audio_bytes, chunk_config)
        if not isinstance(analysis_result, dict):
            raise RuntimeError("Audio analysis returned unexpected result")
        analysis_path = os.path.join(self.run_dir, 'analysis.json')
        self._save_file(analysis_path, json.dumps(analysis_result, indent=2))
        self.asset_paths['analysis'] = analysis_path
        logger.info(f"Analysis saved: {analysis_path}")
        return analysis_result

    def _generate_prompts(self, pacing_chunks: List[Dict]) -> List[Dict]:
        """Step 3: Prompt Generation"""
        logger.info(f"Generating {len(pacing_chunks)} prompts...")
        prompts_data = []
        for i, chunk in enumerate(pacing_chunks):
            try:
                logger.info(f"Generating prompt for chunk {i+1}/{len(pacing_chunks)}...")
                generated_prompt = self.prompt_processor.process(
                    text_chunk=chunk['raw_text'],
                    creative_brief=self.input_data['style_brief']['creative_brief'],
                    global_summary=self.script_content
                )
                if not generated_prompt or generated_prompt.strip() == chunk['raw_text']:
                    logger.warning(f"Prompt generation for chunk {i+1} may have returned a fallback.")
                
                prompts_data.append({
                    'prompt': generated_prompt,
                    'duration_ms': chunk['duration_ms']
                })
            except Exception as e:
                logger.error(f"Failed to generate prompt for chunk {i+1}: {e}")
                raise RuntimeError(f"Prompt generation failed at chunk {i+1}")

        if not prompts_data:
            raise RuntimeError("No prompts were generated.")

        prompts_path = os.path.join(self.run_dir, "prompts.json")
        self._save_file(prompts_path, json.dumps(prompts_data, indent=2))
        self.asset_paths['prompts'] = prompts_path
        logger.info(f"Generated {len(prompts_data)} prompts and saved to {prompts_path}")
        return prompts_data

    def _generate_visuals(self, prompts: List[Dict]):
        """Step 4: Visual Content Generation"""
        logger.info(f"Generating {len(prompts)} visuals...")
        image_sequence = []
        
        direct_fallback = self.config.get('image_generation', {}).get('direct_fallback', False)

        for i, prompt_data in enumerate(prompts):
            image_bytes = None
            logger.info(f"Generating image for chunk {i+1}/{len(prompts)}...")

            if not direct_fallback:
                try:
                    logger.info(f"Attempting to generate image with Gemini for chunk {i+1}...")
                    image_bytes = self.image_generator.process(
                        prompt=prompt_data['prompt'],
                        negative_prompt=self.config['image_generation']['negative_prompt_terms']
                    )
                except Exception as e:
                    logger.warning(f"Gemini image generation failed for chunk {i+1}: {e}. Trying fallback.")

            if image_bytes:
                 logger.info(f"Successfully generated image for chunk {i+1} with Gemini.")
            else:
                if not direct_fallback:
                    logger.info(f"Gemini failed or returned no image for chunk {i+1}. Using fallback.")
                else:
                    logger.info(f"Direct fallback enabled. Using fallback for chunk {i+1}.")
                
                try:
                    bytez_api_key = os.getenv("BYTEZ_API_KEY")
                    image_bytes = generate_image_with_bytez(
                        prompt=prompt_data['prompt'],
                        bytez_api_key=bytez_api_key
                    )
                    if image_bytes:
                        logger.info(f"Successfully generated image for chunk {i+1} with fallback.")
                except Exception as e:
                    logger.error(f"Fallback image generation also failed for chunk {i+1}: {e}")

            if not image_bytes:
                logger.warning(f"Image generation completely failed for chunk {i+1}, skipping.")
                continue

            try:
                img_path = os.path.join(self.image_dir, f"image_{i:03d}.png")
                self._save_file(img_path, image_bytes, mode='wb')
                
                image_sequence.append({
                    'path': img_path,
                    'duration_s': prompt_data['duration_ms'] / 1000.0
                })
            except Exception as e:
                logger.error(f"Failed to save image for visual {i+1}: {e}")
                raise RuntimeError(f"Visual generation failed at chunk {i+1} while saving file.")
        
        if not image_sequence:
            raise RuntimeError("Visual generation failed: No images were created.")
        
        self.asset_paths['image_sequence'] = image_sequence
        image_seq_path = os.path.join(self.run_dir, 'image_sequence.json')
        self._save_file(image_seq_path, json.dumps(image_sequence, indent=2))
        logger.info(f"Successfully generated {len(image_sequence)} images.")

    def _generate_subtitles(self, word_timestamps: List[Dict]):
        """Step 5: Subtitle Generation"""
        logger.info("Generating subtitles...")
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
        self._save_file(subtitle_path, ass_content)
        self.asset_paths['subtitles'] = subtitle_path

    def _render_video(self):
        """Step 6: Final Video Rendering"""
        logger.info("Rendering final video...")
        
        # Prepare audio paths
        main_audio_path = self.asset_paths['master_audio']
        mixed_audio_path = os.path.join(self.temp_dir, "audio_with_background.aac")
        temp_video_path = os.path.join(self.temp_dir, "video_with_audio.mp4")
        final_output_filename = self.input_data['video_details'].get('output_filename', 'final_video.mp4')
        final_output_path = os.path.join(self.run_dir, final_output_filename)

        # Step 1: Mix background music with main audio (if background music directory is provided)
        background_music_dir = self.input_data.get('video_details', {}).get('background_music_path')
        if background_music_dir:
            logger.info("Mixing background music with main audio...")
            bg_music_volume = self.config.get('intro_outro_settings', {}).get('background_music_volume', 0.04)
            
            if not self.video_renderer.mix_background_music(
                main_audio_path=main_audio_path,
                background_music_dir=background_music_dir,
                background_music_volume=bg_music_volume,
                output_path=mixed_audio_path
            ):
                logger.warning("Background music mixing failed. Continuing with main audio only.")
                mixed_audio_path = main_audio_path
        else:
            logger.info("No background music directory provided. Using main audio only.")
            mixed_audio_path = main_audio_path

        # Step 2: Assemble primary video with mixed audio
        if not self.video_renderer.assemble_primary_video(
            image_sequence=self.asset_paths['image_sequence'],
            audio_path=mixed_audio_path,
            render_config=self.config['video_rendering'],
            output_path=temp_video_path
        ):
            raise RuntimeError("Video assembly failed")

        # Step 3: Burn subtitles
        if not self.video_renderer.burn_subtitles(
            video_path=temp_video_path,
            subtitle_path=self.asset_paths['subtitles'],
            output_path=final_output_path
        ):
            raise RuntimeError("Subtitle burn-in failed")

        self.asset_paths['final_video'] = final_output_path
        logger.info(f"Video rendered: {final_output_path}")

    def _cleanup(self):
        """Removes all intermediate files from the output directory, preserving only the final video."""
        if not self.config.get('cleanup_output_dir', False):
            logger.info("Skipping cleanup of output directory as per configuration.")
            return

        logger.info("Cleaning up output directory...")
        
        final_video_path = self.asset_paths.get('final_video')
        if not final_video_path or not os.path.exists(final_video_path):
            logger.warning("Cleanup skipped: Final video path not found or file does not exist.")
            return

        final_video_filename = os.path.basename(final_video_path)

        for item_name in os.listdir(self.run_dir):
            if item_name == final_video_filename:
                logger.debug(f"Preserving final video: {item_name}")
                continue

            item_path = os.path.join(self.run_dir, item_name)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    logger.info(f"Removed directory: {item_path}")
                else:
                    os.remove(item_path)
                    logger.info(f"Removed file: {item_path}")
            except OSError as e:
                logger.warning(f"Error during cleanup of {item_path}: {e}", exc_info=True)
        
        logger.info("Cleanup of output directory complete.")