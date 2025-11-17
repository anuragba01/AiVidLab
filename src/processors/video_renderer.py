"""
Video Renderer Processor Module

This file contains the VideoRenderer class, a powerful tool for assembling the
final video using FFmpeg. It is designed to replicate the complex, multi-stage
rendering logic from the original notebook.

Responsibilities:
- Programmatically construct complex FFmpeg commands.
- Create a primary video assembly from images and audio, including visual
  effects like zoom and crossfades (muxing).
- Mix background music with main audio track.
- Burn subtitles into a video file (hard coding).
- Execute FFmpeg commands and handle success/failure reporting.
"""
import os
import subprocess
import traceback
import math
import random
import shutil
from typing import List, Dict, Any, Optional

class VideoRenderer:
    """
    Assembles the final video using a series of FFmpeg commands.
    """
    def __init__(self):
        print("VideoRenderer initialized.")

    def _execute_ffmpeg_command(self, command: List[str], description: str) -> bool:
        """Helper function to run an FFmpeg command and report status."""
        print(f"Executing FFmpeg command for: {description}...")
        try:
            # For debugging, you can print the full command:
            # print(" ".join(command))
            
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            print(f"Successfully completed: {description}.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR during: {description}. FFmpeg failed.")
            print("--- FFmpeg Command ---")
            print(" ".join(e.cmd))
            print("\n--- FFmpeg Stderr ---")
            print(e.stderr)
            print("\n--- FFmpeg Stdout ---")
            print(e.stdout)
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            return False

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds using ffprobe.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            float: Duration in seconds, or 0.0 if unable to determine.
        """
        try:
            command = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1:noprint_wrappers=1',
                audio_path
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Warning: Could not determine duration of {audio_path}: {e}")
            return 0.0

    def _select_background_music_file(self, music_dir: str) -> Optional[str]:
        """
        Select a random background music file from the specified directory.
        Supports common audio formats: mp3, wav, aac, ogg, m4a, flac.
        
        Args:
            music_dir (str): Path to the directory containing background music files.
            
        Returns:
            Optional[str]: Path to the selected file, or None if no files found.
        """
        if not os.path.isdir(music_dir):
            print(f"Warning: Background music directory not found: {music_dir}")
            return None
        
        supported_formats = ('.mp3', '.wav', '.aac', '.ogg', '.m4a', '.flac')
        music_files = [
            f for f in os.listdir(music_dir) 
            if os.path.isfile(os.path.join(music_dir, f)) and f.lower().endswith(supported_formats)
        ]
        
        if not music_files:
            print(f"Warning: No background music files found in {music_dir}")
            return None
        
        selected_file = random.choice(music_files)
        full_path = os.path.join(music_dir, selected_file)
        print(f"Selected background music: {selected_file}")
        return full_path

    def mix_background_music(
        self,
        main_audio_path: str,
        background_music_dir: str,
        background_music_volume: float,
        output_path: str
    ) -> bool:
        """
        Mix background music with the main audio track.
        
        The background music will be looped if necessary to match the duration of the main audio.
        The volumes are blended: main_audio_volume + background_music_volume (normalized).
        
        Args:
            main_audio_path (str): Path to the main audio (narration/dialogue).
            background_music_dir (str): Path to directory containing background music files.
            background_music_volume (float): Volume level for background music (0.0 to 1.0).
            output_path (str): Path to save the mixed audio file.
            
        Returns:
            bool: True on success, False on failure.
        """
        # Select a random background music file
        bg_music_path = self._select_background_music_file(background_music_dir)
        if not bg_music_path:
            print("Warning: No background music selected. Using main audio only.")
            # If no background music, just copy the main audio
            try:
                shutil.copy(main_audio_path, output_path)
                return True
            except Exception as e:
                print(f"Error copying main audio: {e}")
                return False
        
        # Get duration of main audio
        main_duration = self._get_audio_duration(main_audio_path)
        if main_duration <= 0:
            print(f"Error: Could not determine duration of main audio")
            return False
        
        print(f"Main audio duration: {main_duration:.2f}s")
        
        # Calculate background music volume (0-1 range, where 1 is full volume)
        bg_volume = max(0.0, min(1.0, background_music_volume))
        main_volume = 1.0
        
        # FFmpeg command to mix audio:
        # 1. Loop background music to match main audio duration
        # 2. Adjust volumes
        # 3. Mix both audio streams
        command = [
            'ffmpeg', '-y',
            '-i', main_audio_path,
            '-stream_loop', '-1',  # Loop the background music indefinitely
            '-i', bg_music_path,
            '-filter_complex', (
                f'[0:a]volume={main_volume}[main];'
                f'[1:a]volume={bg_volume}[bg];'
                f'[main][bg]amix=inputs=2:duration=first[out]'
            ),
            '-map', '[out]',
            '-t', str(main_duration),  # Limit output to main audio duration
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]
        
        return self._execute_ffmpeg_command(command, "Background Music Mixing")

    def assemble_primary_video(
        self,
        image_sequence: List[Dict[str, Any]],
        audio_path: str,
        render_config: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Assembles the main video with visuals and audio, but without subtitles.
        This method replicates the zoom, xfade, and concatenation logic.

        Args:
            image_sequence (List[Dict]): A list of dicts, each with 'path' and 'duration_s'.
            audio_path (str): The path to the final mixed audio file.
            render_config (Dict): The 'video_rendering' section from config.json.
            output_path (str): The path to save the intermediate video file.

        Returns:
            bool: True on success, False on failure.
        """
        if not image_sequence:
            print("Error (VideoRenderer): No images provided for assembly.")
            return False

        ffmpeg_inputs = []
        filter_complex_parts = []
        input_map_idx = 0
        
        # --- Build Inputs and Zoom/Pan Filters (Faithful to Notebook Logic) ---
        main_segments_for_xfade = []
        for i, img_data in enumerate(image_sequence):
            # Add image input
            ffmpeg_inputs.extend(['-loop', '1', '-framerate', str(render_config['fps']), '-t', str(img_data['duration_s']), '-i', img_data['path']])
            img_input_label = f"[{input_map_idx}:v]"
            
            # Base processing: scale, pad, set presentation timestamp
            base_img_label = f"[base_img_{i}]"
            filter_complex_parts.append(
                f"{img_input_label}fps={render_config['fps']},"
                f"scale={render_config['target_width']}:{render_config['target_height']}:force_original_aspect_ratio=decrease,"
                f"pad={render_config['target_width']}:{render_config['target_height']}:(ow-iw)/2:(oh-ih)/2,"
                f"setpts=PTS-STARTPTS{base_img_label}"
            )
            
            segment_output_label = base_img_label
            if render_config.get('enable_calm_zoom', False):
                num_frames = int(img_data['duration_s'] * render_config['fps'])
                if num_frames > 0:
                    max_scale = render_config.get('calm_zoom_max_scale', 1.05)
                    cycles = render_config.get('calm_zoom_cycles_per_clip', 0.5)
                    zoom_expr = f"'1+({max_scale}-1)*(0.5-0.5*cos({cycles}*2*PI*on/{num_frames}))'"
                    
                    zoomed_label = f"[zoomed_img_{i}]"
                    filter_complex_parts.append(
                        f"{base_img_label}zoompan=z={zoom_expr}:"
                        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                        f"d={num_frames}:s={render_config['target_width']}x{render_config['target_height']}:fps={render_config['fps']}"
                        f"{zoomed_label}"
                    )
                    segment_output_label = zoomed_label

            main_segments_for_xfade.append(segment_output_label)
            input_map_idx += 1

        # --- Build XFade Chain (Faithful to Notebook Logic) ---
        xfade_duration = render_config.get('transition_duration_s', 1.5)
        if len(main_segments_for_xfade) > 1:
            current_stream = main_segments_for_xfade[0]
            current_duration = image_sequence[0]['duration_s']
            for i in range(len(main_segments_for_xfade) - 1):
                next_stream = main_segments_for_xfade[i+1]
                next_duration = image_sequence[i+1]['duration_s']
                offset = current_duration - xfade_duration
                faded_output = f"[faded_{i}]"
                
                filter_complex_parts.append(
                    f"{current_stream}{next_stream}xfade=transition=fade:duration={xfade_duration}:offset={offset}{faded_output}"
                )
                current_stream = faded_output
                current_duration += next_duration - xfade_duration
            final_visual_stream = current_stream
        else:
            final_visual_stream = main_segments_for_xfade[0]

        # --- Final Command Construction ---
        final_filter_graph = ";".join(filter_complex_parts)
        
        # Add audio input
        ffmpeg_inputs.extend(['-i', audio_path])
        audio_input_idx = input_map_idx

        command = ['ffmpeg', '-y']
        command.extend(ffmpeg_inputs)
        command.extend([
            '-filter_complex', final_filter_graph,
            '-map', f"{final_visual_stream}",
            '-map', f"{audio_input_idx}:a",
            '-c:v', render_config.get('video_codec', 'libx264'),
            '-pix_fmt', render_config.get('pixel_format', 'yuv420p'),
            '-c:a', render_config.get('audio_codec', 'aac'),
            '-shortest', # End when the shortest stream (audio) ends
            output_path
        ])

        return self._execute_ffmpeg_command(command, "Primary Video Assembly")

    def burn_subtitles(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str
    ) -> bool:
        """
        Burns subtitles into an existing video file.

        Args:
            video_path (str): Path to the video with audio (but no subtitles).
            subtitle_path (str): Path to the .ass subtitle file.
            output_path (str): The path for the final output video.

        Returns:
            bool: True on success, False on failure.
        """
        # Escape path for FFmpeg filter syntax, especially for Windows
        escaped_subtitle_path = subtitle_path.replace('\\', '/').replace(':', '\\:')
        
        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f"ass='{escaped_subtitle_path}'",
            '-c:a', 'copy', # Copy audio stream without re-encoding (fast)
            output_path
        ]
        
        return self._execute_ffmpeg_command(command, "Subtitle Burn-in")


