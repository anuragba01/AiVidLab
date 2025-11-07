"""
Video Renderer Processor Module

This file contains the VideoRenderer class, a powerful tool for assembling the
final video using FFmpeg. It is designed to replicate the complex, multi-stage
rendering logic from the original notebook.

Responsibilities:
- Programmatically construct complex FFmpeg commands.
- Create a primary video assembly from images and audio, including visual
  effects like zoom and crossfades (muxing).
- Burn subtitles into a video file (hard coding).
- Execute FFmpeg commands and handle success/failure reporting.
"""
import os
import subprocess
import traceback
import math
from typing import List, Dict, Any

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


# --- Independent Test Block ---
if __name__ == '__main__':
    print("\n--- Running Independent Test for VideoRenderer ---")
    
    # To run this test, you need mock assets:
    # - Two dummy images (e.g., 'test_image1.png', 'test_image2.png')
    # - One dummy audio file (e.g., 'test_audio.aac')
    # - One dummy subtitle file (e.g., 'test_subtitles.ass')
    
    print("This test requires mock asset files to be created.")
    # Create dummy assets for the test
    try:
        from PIL import Image
        Image.new('RGB', (100, 100), color = 'red').save('test_image1.png')
        Image.new('RGB', (100, 100), color = 'blue').save('test_image2.png')
        # A simple silent AAC requires ffmpeg, so we'll skip its creation here
        # Assuming a silent audio file `test_audio.aac` exists.
        # Create a dummy silent audio file with ffmpeg if it doesn't exist.
        if not os.path.exists('test_audio.aac'):
            subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo', '-t', '8', '-c:a', 'aac', 'test_audio.aac'], check=True)
        
        with open('test_subtitles.ass', 'w') as f:
            f.write("[Script Info]\nTitle: Test\n[V4+ Styles]\nStyle: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello World")

        print("Mock assets created successfully.")
        
        mock_image_seq = [
            {'path': 'test_image1.png', 'duration_s': 5},
            {'path': 'test_image2.png', 'duration_s': 5}
        ]
        mock_audio_path = 'test_audio.aac'
        mock_subtitle_path = 'test_subtitles.ass'
        
        # From config.json
        mock_render_config = {
            "target_width": 1280,
            "target_height": 720,
            "fps": 24,
            "transition_duration_s": 1.0,
            "video_codec": "libx264",
            "pixel_format": "yuv420p",
            "audio_codec": "aac",
            "enable_calm_zoom": True,
            "calm_zoom_max_scale": 1.1,
            "calm_zoom_cycles_per_clip": 1
        }
        
        temp_video_path = "test_temp_video.mp4"
        final_video_path = "test_final_video.mp4"

        renderer = VideoRenderer()
        
        print("\n--- Testing Step 1: Primary Assembly ---")
        success1 = renderer.assemble_primary_video(mock_image_seq, mock_audio_path, mock_render_config, temp_video_path)
        
        if success1 and os.path.exists(temp_video_path):
            print("SUCCESS: Primary assembly seems to have worked.")
            
            print("\n--- Testing Step 2: Subtitle Burn-in ---")
            success2 = renderer.burn_subtitles(temp_video_path, mock_subtitle_path, final_video_path)
            
            if success2 and os.path.exists(final_video_path):
                print(f"\nSUCCESS: Final video with subtitles saved to '{final_video_path}'")
            else:
                print("\nFAILURE: Subtitle burn-in failed.")
        else:
            print("\nFAILURE: Primary assembly failed.")

    except Exception as e:
        print(f"\nAn error occurred during the test setup or execution: {e}")