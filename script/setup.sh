#!/bin/bash
# -----------------------------------------------------------------------------
# Setup Script for the Automated Video Pipeline
# -----------------------------------------------------------------------------
# This script installs all necessary system and Python dependencies.
#
# To run this script:
# 1. Make it executable:  chmod +x setup.sh
# 2. Run it:             ./setup.sh
# -----------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Environment Setup ---"

# --- 1. Install System Dependencies (for Debian/Ubuntu) ---
# FFmpeg is essential for the VideoRenderer.
echo "\n>>> Step 1: Installing system dependencies (FFmpeg)..."
sudo apt-get update -y
sudo apt-get install -y ffmpeg

echo "FFmpeg installed successfully."

# --- 2. Install Python Dependencies ---
echo "\n>>> Step 2: Installing Python libraries..."

# Check if a NVIDIA GPU is available to decide how to install PyTorch
if command -v nvidia-smi &> /dev/null
then
    echo "NVIDIA GPU detected! Installing PyTorch with CUDA support."
    # This is the recommended command for installing PyTorch with CUDA 12.1
    # It ensures Whisper can use the GPU for fast transcription.
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected. Installing the standard CPU version of PyTorch."
    # The requirements.txt file will fetch the standard CPU version.
fi

echo "Installing remaining Python packages from requirements.txt..."
pip3 install -r requirements.txt

echo "All Python libraries installed successfully."

# --- 3. Information about AI Model Downloads ---
echo "\n>>> Step 3: AI Model Information"
echo "NOTE: The Whisper speech-to-text model will be downloaded automatically the FIRST TIME the AudioAnalyzer is run."
echo "This is a one-time download (per model size) and the model will be cached for future runs."
echo "The download size can be several hundred MB to over a GB depending on the model size set in config.json."

echo "\n--- Environment Setup Complete! ---"