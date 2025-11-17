#!/usr/bin/env python3
import subprocess
import sys
import platform
import shutil


def run(cmd):
    try:
        subprocess.run(cmd, check=True, shell=True)
    except Exception:
        sys.exit(1)


def ffmpeg_installed():
    return shutil.which("ffmpeg") is not None


def install_ffmpeg():
    os_type = platform.system()

    if os_type == "Linux":
        run("sudo apt-get update -y")
        run("sudo apt-get install -y ffmpeg")

    elif os_type == "Darwin":
        if shutil.which("brew"):
            run("brew install ffmpeg")
        else:
            print("Install Homebrew first: https://brew.sh")
            sys.exit(1)

    elif os_type == "Windows":
        print("Install FFmpeg manually: https://ffmpeg.org/download.html")
        sys.exit(1)


def install_pytorch_cpu():
    run("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")


def install_python_packages():
    packages = [
        "google-genai",
        "bytez",
        "pydub",
        "Pillow",
        "python-dotenv",
        "git+https://github.com/linto-ai/whisper-timestamped.git"
    ]

    for p in packages:
        run(f"pip install {p}")


def main():
    print("Checking FFmpeg...")
    if not ffmpeg_installed():
        install_ffmpeg()

    print("Installing PyTorch (CPU only)...")
    install_pytorch_cpu()

    print("Installing Python packages...")
    install_python_packages()

    print("Setup complete.")


if __name__ == "__main__":
    main()
