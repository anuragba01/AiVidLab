import os
import json
import pytest
import shutil
from src.orchestrator import Orchestrator

# Define a temporary directory for test outputs
TEST_OUTPUT_DIR = "test_e2e_output"

@pytest.fixture(scope="module")
def setup_teardown_test_environment():
    """
    A pytest fixture to set up the testing environment before tests run
    and tear it down afterward.
    """
    # --- Setup ---
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    # Create a mock config file for testing
    mock_config = {
        "directories": {
            "output": TEST_OUTPUT_DIR,
            "images": f"{TEST_OUTPUT_DIR}/images",
            "audio": f"{TEST_OUTPUT_DIR}/audio",
            "temp": f"{TEST_OUTPUT_DIR}/temp"
        },
        "gemini_models": {
            "tts": "gemini-1.5-flash",
            "m_llm": "gemini-1.5-flash",
            "llm": "gemini-1.5-flash",
            "image_generator": "gemini-1.5-flash"
        },
        "tts_settings": {"voice_name": "Echo", "audio_rate_hz": 24000},
        "audio_analysis": {
            "min_chunk_duration_s": 2.0,
            "max_chunk_duration_s": 5.0,
            "min_silence_len_ms": 300
        },
        "image_generation": {
            "negative_prompt_terms": "low quality, text, signature"
        },
        "video_rendering": {
            "target_width": 1280,
            "target_height": 720,
            "fps": 24,
            "transition_duration_s": 0.5,
            "video_codec": "libx264",
            "pixel_format": "yuv420p",
            "audio_codec": "aac",
            "enable_calm_zoom": False
        },
        "subtitle_style": {
            "default": {
                "font_name": "Arial", "font_size": 48, "primary_colour": "&H00FFFFFF",
                "outline_colour": "&H00000000", "back_colour": "&H80000000", "alignment": 2
            },
            "line_rules": {"max_words_per_line": 5}
        },
        "cleanup_output_dir": False  # Disable cleanup to inspect files
    }
    
    config_path = os.path.join(TEST_OUTPUT_DIR, "test_config.json")
    with open(config_path, 'w') as f:
        json.dump(mock_config, f, indent=2)

    # Create a mock input file for testing
    mock_input = {
        "script_generation": {
            "topics": ["The concept of 'flow state'"],
            "keywords": ["psychology", "focus", "productivity"],
            "tone": "concise and educational",
            "target_word_count": 30
        },
        "video_details": {
            "output_filename": "e2e_test_video.mp4"
        },
        "style_brief": {
            "creative_brief": "Style: Minimalist, clean visuals. Mood: Calm and focused."
        }
    }
    input_path = os.path.join(TEST_OUTPUT_DIR, "test_input.json")
    with open(input_path, 'w') as f:
        json.dump(mock_input, f, indent=2)

    # Yield the paths to the test function
    yield config_path, input_path

    # --- Teardown ---
    # shutil.rmtree(TEST_OUTPUT_DIR)

def test_e2e_pipeline(setup_teardown_test_environment):
    """
    Runs a full end-to-end test of the video generation pipeline.
    """
    # Get paths from the fixture
    config_path, input_path = setup_teardown_test_environment

    # --- Execute ---
    # Ensure API key is available
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY environment variable not set. Skipping E2E test.")

    orchestrator = Orchestrator(
        config_path=config_path,
        input_path=input_path,
    )
    success = orchestrator.run_pipeline()

    # --- Assert ---
    # 1. Check that the pipeline reported success
    assert success, "Orchestrator.run_pipeline() returned False."

    # 2. Check for the final video file
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    output_filename = input_data['video_details']['output_filename']
    final_video_path = os.path.join(TEST_OUTPUT_DIR, output_filename)
    
    assert os.path.exists(final_video_path), f"Final video file was not created at {final_video_path}"
    assert os.path.getsize(final_video_path) > 0, "Final video file is empty."

    # 3. Check for key intermediate files
    expected_files = [
        "generated_script.txt",
        "master_audio.wav",
        "analysis.json",
        "prompts.json",
        "image_sequence.json",
        "subtitles.ass",
    ]
    for filename in expected_files:
        path = os.path.join(TEST_OUTPUT_DIR, filename)
        assert os.path.exists(path), f"Intermediate file '{filename}' was not created."
        assert os.path.getsize(path) > 0, f"Intermediate file '{filename}' is empty."

    # 4. Check that at least one image was generated
    image_dir = os.path.join(TEST_OUTPUT_DIR, "images")
    assert os.path.isdir(image_dir), "Images directory was not created."
    images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    assert len(images) > 0, "No images were generated in the images directory."
