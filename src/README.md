# Project Technical Architecture & Data Flow

This document provides a detailed technical breakdown of the AiVidLab pipeline, focusing on the software architecture, data flow between components, and the technologies used. It is intended for developers looking to understand the internal workings of the system.

## 1. Core Architectural Principles

-   **Orchestration Pattern**: The system is orchestrated by the `Orchestrator` class (`src/orchestrator.py`), which acts as a state machine. It manages the entire video creation workflow by sequentially invoking a series of "processor" modules. The `PipelineStage` enum is used to track the current state, which facilitates error handling and allows for the potential resumption of failed runs by skipping already completed stages.

-   **Processor Modularity**: Each distinct task in the pipeline (e.g., script generation, audio analysis) is encapsulated within its own class in the `src/processors/` directory. This modular design promotes separation of concerns, making the system easier to debug, test, and extend. For instance, the underlying TTS model can be changed by modifying only `TTSProcessor` without affecting the rest of the pipeline.

-   **Configuration-Driven Workflow**: The pipeline's behavior is controlled by two distinct JSON files:
    -   `config.json`: Contains low-level technical parameters for the application, such as API model names (`gemini-2.5-pro`, `gemini-2.5-flash-image`), rendering settings (dimensions, codecs, FPS), and styling rules for subtitles.
    -   `input.json`: Provides the high-level creative brief for a single video generation run, including the topics to cover, specific keywords, and the desired tone.

## 2. Detailed Pipeline Stages & Data Flow

The `Orchestrator` executes the following stages in sequence. Each stage's output serves as the input for subsequent stages.

1.  **Script Generation (`ScriptGenerator`)**
    -   **Input**: A dictionary from `input.json` containing `topics` (List[str]), `keywords` (List[str]), and `tone` (str).
    -   **Process**: Constructs a prompt for a Gemini LLM via the `google-genai` library to generate a coherent script.
    -   **Output**: A single string containing the script content. This is stored in the `Orchestrator.script_content` attribute and cached on disk as `generated_script.txt`.

2.  **Audio Generation (`TTSProcessor`)**
    -   **Input**: The `script_content` string.
    -   **Process**: Uses a Gemini TTS model through the `google-genai` library to synthesize speech from the script text.
    -   **Output**: Raw audio bytes, which are then saved as `master_audio.wav`. The path is stored in `Orchestrator.asset_paths['master_audio']`.

3.  **Audio Analysis (`AudioAnalyzer`)**
    -   **Input**: The file path to `master_audio.wav`.
    -   **Process**: Leverages the `whisper-timestamped` library to perform speech-to-text and obtain word-level timestamps. It then applies a custom chunking algorithm based on silence detection and duration thresholds (from `config.json`) to group words into semantically coherent "pacing chunks" that dictate scene timing.
    -   **Output**: A JSON object written to `analysis.json`. This object contains two keys:
        -   `word_timestamps`: A `List[Dict]` where each dict is `{'text': str, 'start': float, 'end': float}`.
        -   `pacing_chunks`: A `List[Dict]` where each dict is `{'raw_text': str, 'duration_ms': int}`.

4.  **Prompt Generation (`PromptProcessor`)**
    -   **Input**: The `pacing_chunks` list from the previous stage and the `creative_brief` string from `input.json`.
    -   **Process**: Iterates through each chunk in `pacing_chunks`. For each, it constructs a new, detailed prompt for a Gemini LLM, combining the chunk's text with the global `creative_brief` to generate a visually descriptive prompt suitable for an image generator.
    -   **Output**: A `List[Dict]` where each dict contains a `prompt` string and its associated `duration_ms`. This is saved to `prompts.json` and passed in memory to the next stage.

5.  **Visual Generation (Logic in `Orchestrator`)**
    -   **Input**: The list of prompt dictionaries.
    -   **Process**: The core logic resides within `Orchestrator._generate_visuals`. It reads the `direct_fallback` boolean from `config.json`.
        -   If `False` (default), it calls `ImageGenerator`, which is a wrapper around the Gemini image generation API.
        -   If the primary call fails or if `direct_fallback` is `True`, the orchestrator directly calls the `generate_image_with_bytez` utility, which uses the `bytez` library to interface with the Stability AI API.
    -   **Output**: A sequence of PNG images saved to `output/images/`. A `List[Dict]` containing the file `path` and `duration_s` for each image is compiled for the rendering stage.

6.  **Subtitle Generation (`SubtitleProcessor`)**
    -   **Input**: The `word_timestamps` list and styling parameters from `config.json` (`subtitle_style`, `heading_style`).
    -   **Process**: This module constructs an Advanced SubStation Alpha (`.ass`) subtitle file.
        -   It uses Python's `difflib` for fuzzy string matching to locate headings within the transcript.
        -   It generates `.ass` style definitions and then creates `Dialogue` entries for each line. Word timestamps are used to group words into lines based on configured constraints like `max_words_per_line`.
    -   **Output**: A single string containing the complete `.ass` file content, saved to `subtitles.ass`.

7.  **Video Rendering (`VideoRenderer`)**
    -   **Input**: A collection of all asset paths and the `video_rendering` section from `config.json`.
    -   **Process**: This class acts as a wrapper for the `ffmpeg` command-line tool, building and executing commands via Python's `subprocess` module.
        -   **Audio Mixing**: Uses the `amix` audio filter to combine the narration and background music.
        -   **Visual Assembly**: Programmatically builds a complex `-filter_complex` graph to chain all image inputs. It uses the `zoompan` filter for smooth zooming/panning effects and the `xfade` filter for crossfade transitions.
        -   **Subtitle Burn-in**: Uses the `ass` video filter to hardcode the subtitles directly onto the video frames.
    -   **Output**: The final, rendered `final_video.mp4` file.

## 3. Core Component Technologies

-   **`src/main.py`**: Application entry point.
-   **`src/orchestrator.py`**: Central pipeline controller.
-   **`src/processors/`**:
    -   `script_generator.py`: `google-genai`
    -   `tts_processor.py`: `google-genai`
    -   `audio_analyzer.py`: `whisper-timestamped`
    -   `prompt_processor.py`: `google-genai`
    -   `image_generator.py`: `google-genai`
    -   `subtitle_processor.py`: `difflib`
    -   `video_renderer.py`: `subprocess` (for `ffmpeg`)
-   **`src/utilities/`**:
    -   `fallback_image_generator.py`: `bytez` (for Stability AI), `requests`
    -   `image_utils.py`: `Pillow`
    -   `json_load.py`: `json`, `os`
-   **External Dependencies**:
    -   **FFmpeg**: Core multimedia framework for all rendering tasks.
    -   **Gemini API**: Primary AI provider for text, speech, and image generation.
    -   **Bytez/Stability AI API**: Fallback provider for image generation.