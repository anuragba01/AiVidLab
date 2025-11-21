<div align="center">

# AiVidLab

**An automated pipeline for video creation.**

</div>

---

## Overview

AiVidLab is an end-to-end pipeline for generating videos automatically of **`any lenght`**. By providing a topic and some initial instructions, this tool leverages Large Language Models (LLMs) and other AI services to produce a complete video. The pipeline handles everything from scriptwriting to final rendering, making it an ideal solution for automating video content creation.

The core functionalities include:
- Script Generation
- Image Generation
- Text-to-Speech (TTS) Conversion
- Video Rendering
- Burnt-in Subtitles
- Background Music

---

## How AiVidLab Works

The video generation process is orchestrated as a pipeline, with each step feeding into the next:

1.  **Prompt Processing:** The initial topic is processed to generate a detailed prompt for script generation.
2.  **Script Generation:** An LLM generates a script based on the processed prompt.
3.  **Image Generation:** AI models create images based on the script's content.
4.  **Text-to-Speech (TTS):** The script is converted into audio.
5.  **Audio Analysis:** The audio is analyzed to determine timings for subtitles and scene changes.
6.  **Subtitle Processing:** Subtitles are generated and timed to match the audio.
7.  **Video Rendering:** The images, audio, and subtitles are combined to render the final video.

---
## Architect of AiVidLab
 ![Architect of AiVidLab](https://github.com/user-attachments/assets/efd57e8e-369b-4629-ab29-f460dbf7eb2c)

## Getting Started

### Prerequisites

This project requires the following to be installed on your system:

*   **Python (tested on v3.12)**
*   **FFmpeg:** A command-line tool for handling multimedia data. The installation script will attempt to install it for you on Linux and macOS. For Windows, you will need to install it manually.

The project also depends on the following Python packages, which will be installed by the setup script:
*   `torch`, `torchvision`, `torchaudio` (CPU version)
*   `google-genai`
*   `bytez`
*   `pydub`
*   `Pillow`
*   `python-dotenv`
*   `whisper-timestamped`

`Note:` Software has only tested tested on **ubuntu-latest**, **windows-latest** and **macos-latest**.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/anuragba01/AiVidLab.git
    cd AiVidLab
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Run the setup script:**
    This script will install all the necessary dependencies and create `.env` file.
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

### Configuration

The setup script will create a `.env` file in the project's root directory. You need to add your API keys to this file:

```
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
BYTEZ_API_KEY=YOUR_BYTEZ_API_KEY
```

### Usage

1.  **Edit `input.json`:**
Define the topic and instructions for your video. <br>
**Edit** `config.json` if needed  and set your configaration.


2.  **Run the pipeline:**
    ```bash
    python -m src.main.py
    ```
3.  **Find your video:** The generated video will be saved in the `output` directory.<br>

⚠️**Cost Warning: Image Generation**
By default, the software is configured to use **gemini-2.5-flash-image** (a *Paid API*).  
If Gemini fails, the system will automatically fall back to **stabilityai/stable-diffusion-xl-base-1.0** (Free/bytez). <br>
**Note**: image generation with free api can be paifully slow.

To avoid API costs entirely, you can force the system to use the free model immediately by updating your config.

In **config.json**:

```bash
"image_generator": {
    "direct_fallback": true
}
```
---
## How to get APIs key
Visit [GEMINI_API_KEY](https://ai.google.dev/gemini-api/docs/api-key) for *`GEMINI_API_KEY`*
<br>
Visit [BYTEZ_API_KEY](https://bytez.com) for 
*`BYTEZ_API_KEY`*

## Testing
Please visit the **`.github/workflows`** folder for information related to testing.

## Contributing

Contributions are welcome! If you have any ideas or suggestions please open an issue or submit a pull request.

---

## License
Visit licence section to know about licence.
