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
import io
import logging
from enum import Enum
from typing import Dict, List, Any, Optional
from pydub import AudioSegment

from processors.script_generator import ScriptGenerator
from processors.tts_processor import TTSProcessor
from processors.audio_analyzer import AudioAnalyzer
from processors.prompt_processor import PromptProcessor
from processors.image_generator import ImageGenerator
from processors.subtitle_processor import SubtitleProcessor
from processors.video_renderer import VideoRenderer

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages for tracking and recovery."""
    INIT = "initialization"
    SCRIPT = "script_generation"
    AUDIO = "audio_generation"
    ANALYSIS = "audio_analysis"
    VISUALS = "visual_generation"
    SUBTITLES = "subtitle_generation"
    RENDER = "video_rendering"
    COMPLETE = "complete"


class Orchestrator:
    """Manages video creation workflow with proper error handling and state tracking."""

    def __init__(self, config_path: str, input_path: str):
        self.config = self._load_json(config_path)
        self.input_data = self._load_json(input_path)
        # Use a fixed output directory (no timestamp) so generated filenames are stable
        # across runs. We'll clear the output directory at startup so each run starts
        # from a clean slate unless files are placed there manually.
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
        # Use configured output directory directly (no timestamp)
        self.run_dir = os.path.abspath(self.config['directories']['output'])
        self.image_dir = os.path.join(self.run_dir, "images")
        self.audio_dir = os.path.join(self.run_dir, "audio")
        self.temp_dir = os.path.join(self.run_dir, "temp")
        # Ensure the output directory exists and is empty (remove existing contents)
        os.makedirs(self.run_dir, exist_ok=True)
        # ensure subdirectories exist so later writes won't fail
        for d in (self.image_dir, self.audio_dir, self.temp_dir):
            os.makedirs(d, exist_ok=True)



    def _initialize_processors(self):
        """Initializes all processor modules."""
        try:
            self.script_generator = ScriptGenerator(self.config['gemini_models']['llm'])
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
            (PipelineStage.VISUALS, self._generate_visuals),
            (PipelineStage.SUBTITLES, self._generate_subtitles),
            (PipelineStage.RENDER, self._render_video),
        ]

        
        

        try:
            analysis_result = None
            
            for stage, step_func in pipeline_steps:
                self.current_stage = stage
                logger.info(f"Executing stage: {stage.value}")

                # After each step, skip it if the expected output already exists
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
                    # analysis can be cached to analysis.json
                    analysis_path = os.path.join(self.run_dir, 'analysis.json')
                    if os.path.exists(analysis_path):
                        logger.info("Analysis already exists, loading cached analysis")
                        with open(analysis_path, 'r', encoding='utf-8') as f:
                            analysis_result = json.load(f)
                    else:
                        analysis_result = step_func()

                elif stage == PipelineStage.VISUALS:
                    # visuals produce image files and an image_sequence.json
                    image_seq_path = os.path.join(self.run_dir, 'image_sequence.json')
                    if os.path.exists(image_seq_path):
                        logger.info("Image sequence already exists, skipping visual generation")
                        with open(image_seq_path, 'r', encoding='utf-8') as f:
                            image_sequence = json.load(f)
                        self.asset_paths['image_sequence'] = image_sequence
                    else:
                        step_func(analysis_result['pacing_chunks'])

                elif stage == PipelineStage.SUBTITLES:
                    subtitle_path = os.path.join(self.run_dir, 'subtitles.ass')
                    if os.path.exists(subtitle_path):
                        logger.info("Subtitles already exist, skipping subtitle generation")
                        self.asset_paths['subtitles'] = subtitle_path
                    else:
                        step_func(analysis_result['word_timestamps'])

                elif stage == PipelineStage.RENDER:
                    final_output_filename = self.input_data['video_details'].get(
                        'output_filename', 'final_video.mp4'
                    )
                    final_output_path = os.path.join(self.run_dir, final_output_filename)
                    if os.path.exists(final_output_path):
                        logger.info("Final video already exists, skipping rendering")
                        self.asset_paths['final_video'] = final_output_path
                    else:
                        step_func()

                else:
                    step_func()
                
                self._save_state()

            self.current_stage = PipelineStage.COMPLETE
            duration = time.time() - start_time

            logger.info(f"Pipeline completed successfully in {duration:.2f}s")
            logger.info(f"Final video: {self.asset_paths.get('final_video', 'UNKNOWN')}")

            # Run cleanup now that the pipeline finished successfully
            try:
                cleanup = getattr(self, "_cleanup", None)
                if callable(cleanup):
                    cleanup()
            except Exception:
                logger.exception("Cleanup failed after successful run (non-critical)")

            return True

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Pipeline failed at {self.current_stage.value} after {duration:.2f}s")
            logger.error(f"Error: {e}")
            logger.debug(traceback.format_exc())
            return False

        finally:
            # Cleanup logic removed per request (no-op)
            pass

    # Cleanup logic removed intentionally.

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
        """Step 2: Audio analysis — saves analysis.json and pacing_chunks.json"""
        logger.info("Analyzing audio...")

        master_audio_path = self.asset_paths.get('master_audio') or os.path.join(self.audio_dir, "master_audio.wav")
        if not os.path.exists(master_audio_path):
            raise FileNotFoundError(f"Master audio not found: {master_audio_path}")

        # Call the audio analyzer. The current AudioAnalyzer expects (audio_bytes, chunk_config).
        chunk_config = self.config.get('audio_analysis', {})
        try:
            with open(master_audio_path, 'rb') as f:
                audio_bytes = f.read()

            analysis_result = self.audio_analyzer.process(audio_bytes, chunk_config)
        except TypeError:
            # Backwards-compatible fallbacks: try path then AudioSegment with chunk_config
            try:
                analysis_result = self.audio_analyzer.process(master_audio_path, chunk_config)
            except TypeError:
                audio_seg = AudioSegment.from_file(master_audio_path)
                analysis_result = self.audio_analyzer.process(audio_seg, chunk_config)

        if not isinstance(analysis_result, dict):
            raise RuntimeError("Audio analysis returned unexpected result")

        # Persist full analysis
        analysis_path = os.path.join(self.run_dir, 'analysis.json')
        try:
            self._save_file(analysis_path, json.dumps(analysis_result, indent=2))
            self.asset_paths['analysis'] = analysis_path
            logger.info(f"Analysis saved: {analysis_path}")
        except Exception:
            logger.warning("Failed to save analysis.json (non-critical)")

        # Persist pacing chunks separately for easy access
        pacing_chunks = analysis_result.get('pacing_chunks', [])
        pacing_path = os.path.join(self.run_dir, 'pacing_chunks.json')
        try:
            self._save_file(pacing_path, json.dumps(pacing_chunks, indent=2))
            self.asset_paths['pacing_chunks'] = pacing_path
            logger.info(f"Pacing chunks saved: {pacing_path} ({len(pacing_chunks)} items)")
        except Exception:
            logger.warning("Failed to save pacing_chunks.json (non-critical)")

        return analysis_result

    
    
    def _generate_visuals(self, pacing_chunks: List[Dict]):
        """Step 3: Visual Content Generation"""
        logger.info(f"Generating {len(pacing_chunks)} visuals...")
        image_sequence = []
        
        
        for i, chunk in enumerate(pacing_chunks):
            try:
                # Generate an image-specific prompt from the chunk using the PromptProcessor.
                # The prompt processor is intentionally responsible only for prompt creation.
                prompt = self.prompt_processor.generate_image_prompt(
                    text_chunk=chunk['raw_text'],
                    creative_brief=self.input_data['style_brief']['creative_brief'],
                    global_summary=self.script_content
                )

                # Pass the generated prompt to the ImageGenerator which handles model-specific
                # API details and negative prompts.
                image_bytes = self.image_generator.process(
                    prompt=prompt,
                    negative_prompt=self.config['image_generation']['negative_prompt_terms']
                )
                
                if not image_bytes:
                    logger.warning(f"Empty image for chunk {i+1}, skipping")
                    continue
                
                img_path = os.path.join(self.image_dir, f"image_{i:03d}.png")
                self._save_file(img_path, image_bytes, mode='wb')
                
                image_sequence.append({
                    'path': img_path,
                    'duration_s': chunk['duration_ms'] / 1000.0
                })
                
            except Exception as e:
                logger.error(f"Failed to generate visual {i+1}: {e}")
                raise RuntimeError(f"Visual generation failed at chunk {i+1}")
        
        if not image_sequence:
            raise RuntimeError("No images generated")
        
        self.asset_paths['image_sequence'] = image_sequence
        # persist image sequence for skip-on-restart
        try:
            image_seq_path = os.path.join(self.run_dir, 'image_sequence.json')
            with open(image_seq_path, 'w', encoding='utf-8') as f:
                json.dump(image_sequence, f, indent=2)
            self.asset_paths['image_sequence'] = image_sequence
        except Exception:
            logger.warning("Failed to persist image_sequence.json (non-critical)")
        logger.info(f"Generated {len(image_sequence)} images")

    def _generate_subtitles(self, word_timestamps: List[Dict]):
        """Step 4: Subtitle Generation"""
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
        """Step 5: Final Video Rendering"""
        logger.info("Rendering final video...")
        
        temp_video_path = os.path.join(self.temp_dir, "video_with_audio.mp4")
        final_output_filename = self.input_data['video_details'].get(
            'output_filename',
            'final_video.mp4'
        )
        final_output_path = os.path.join(self.run_dir, final_output_filename)

        if not self.video_renderer.assemble_primary_video(
            image_sequence=self.asset_paths['image_sequence'],
            audio_path=self.asset_paths['master_audio'],
            render_config=self.config['video_rendering'],
            output_path=temp_video_path
        ):
            raise RuntimeError("Video assembly failed")

        if not self.video_renderer.burn_subtitles(
            video_path=temp_video_path,
            subtitle_path=self.asset_paths['subtitles'],
            output_path=final_output_path
        ):
            raise RuntimeError("Subtitle burn-in failed")

        self.asset_paths['final_video'] = final_output_path
        logger.info(f"Video rendered: {final_output_path}")

    """def _cleanup(self):


        The cleanup implementation is preserved but moved to the end of the file
        so the function definition appears last. The orchestrator currently does
        not call this automatically (the finally block was left as a no-op by
        request), but the method remains available for manual invocation.
        
        if not self.config.get('cleanup_temp_dir', True):
            return

        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("Temporary directory cleaned")
            except OSError as e:
                logger.warning(f"Cleanup failed (non-critical): {e}")"""