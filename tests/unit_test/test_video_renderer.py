import pytest
from src.processors.video_renderer import VideoRenderer
import subprocess
import os
from PIL import Image


def test_video_rendering():
    print("\n--- Running Independent Test for VideoRenderer ---")
    
    # To run this test, you need mock assets:
    # - Two dummy images (e.g., 'test_image1.png', 'test_image2.png')
    # - One dummy audio file (e.g., 'test_audio.aac')
    # - One dummy subtitle file (e.g., 'test_subtitles.ass')
    
    print("This test requires mock asset files to be created.")
    
    temp_video_path = "test_temp_video.mp4"
    final_video_path = "test_final_video.mp4"
    mock_files = ['test_image1.png', 'test_image2.png', 'test_audio.aac', 'test_subtitles.ass', temp_video_path, final_video_path]

    try:
        # Create dummy assets for the test
        Image.new('RGB', (100, 100), color = 'red').save('test_image1.png')
        Image.new('RGB', (100, 100), color = 'blue').save('test_image2.png')
        
        if not os.path.exists('test_audio.aac'):
            subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo', '-t', '8', '-c:a', 'aac', 'test_audio.aac'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with open('test_subtitles.ass', 'w') as f:
            f.write("[Script Info]\nTitle: Test\n[V4+ Styles]\nStyle: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello World")

        print("Mock assets created successfully.")
        
        mock_image_seq = [
            {'path': 'test_image1.png', 'duration_s': 5},
            {'path': 'test_image2.png', 'duration_s': 5}
        ]
        mock_audio_path = 'test_audio.aac'
        mock_subtitle_path = 'test_subtitles.ass'
        
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
        
        renderer = VideoRenderer()
        
        print("\n--- Testing Step 1: Primary Assembly ---")
        success1 = renderer.assemble_primary_video(mock_image_seq, mock_audio_path, mock_render_config, temp_video_path)
        assert success1, "Primary assembly method returned False."
        assert os.path.exists(temp_video_path), "Primary assembly did not create the temp video file."
        print("SUCCESS: Primary assembly seems to have worked.")
        
        print("\n--- Testing Step 2: Subtitle Burn-in ---")
        success2 = renderer.burn_subtitles(temp_video_path, mock_subtitle_path, final_video_path)
        assert success2, "Subtitle burn-in method returned False."
        assert os.path.exists(final_video_path), "Subtitle burn-in did not create the final video file."
        print(f"\nSUCCESS: Final video with subtitles saved to '{final_video_path}'")

    except Exception as e:
        pytest.fail(f"An error occurred during the test setup or execution: {e}")
    finally:
        # Clean up mock files
        print("\nCleaning up mock files...")
        for f in mock_files:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed {f}")


if __name__ == '__main__':
    test_video_rendering()