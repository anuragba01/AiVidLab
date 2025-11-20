from src.processors.video_renderer import VideoRenderer
import subprocess
import os
from PIL import Image


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