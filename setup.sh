#!/bin/bash

REQUIREMENTS_FILE="requirements.txt"
ENV_FILE=".env"

echo "Starting project setup..."

# 1. Check and Install FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is already installed."
else
    echo "FFmpeg not found. Attempting installation..."
    OS="$(uname -s)"
    case "${OS}" in
        Linux*)
            if [ -f /etc/debian_version ]; then
                sudo apt-get update && sudo apt-get install -y ffmpeg
            else
                echo "Manual FFmpeg installation required for this Linux distribution."
                exit 1
            fi
            ;;
        Darwin*)
            if command -v brew &> /dev/null; then
                brew install ffmpeg
            else
                echo "Homebrew not found. Install Homebrew or FFmpeg manually."
                exit 1
            fi
            ;;
        *)
            echo "Automatic installation not supported for this OS. Please install FFmpeg manually."
            # We continue to Python setup even if FFmpeg fails here
            ;;
    esac
fi

# 2. Install Python Dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    python3 -m pip install --upgrade pip
    python3 -m pip install -r "$REQUIREMENTS_FILE"
    
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies."
        exit 1
    fi
else
    echo "Error: $REQUIREMENTS_FILE not found."
    exit 1
fi

# 3. Setup Environment Variables
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating $ENV_FILE..."
    printf "GEMINI_API_KEY=\nBYTEZ_API_KEY=\nLOG_LEVEL=INFO\n" > "$ENV_FILE"
    echo "Created $ENV_FILE. Please update it with your API keys."
else
    echo "$ENV_FILE already exists. Skipping."
fi

echo "Setup complete."