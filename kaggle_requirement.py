#!/usr/bin/env python3
import subprocess
import sys


def run(cmd):
    try:
        subprocess.run(cmd, check=True, shell=True)
    except Exception:
        sys.exit(1)


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
    print("Installing Python packages for Kaggle...")
    install_python_packages()
    print("Kaggle setup complete.")


if __name__ == "__main__":
    main()

