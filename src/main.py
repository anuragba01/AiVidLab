"""
Main Entry Point

This is the main script to run the entire video generation pipeline.
It sets up the Orchestrator with the paths to the configuration and
user input files and then starts the process.
"""
import sys

from orchestrator import Orchestrator

# Define the paths to your configuration and input files.
# In a real application, you might get these from command-line arguments.
CONFIG_FILE_PATH = "config.json"
USER_INPUT_FILE_PATH = "user_input.json"

if __name__ == "__main__":
    
    # Optional: Allow passing the user input file as a command-line argument
    # Example: python main.py my_other_video_input.json
    if len(sys.argv) > 1:
        USER_INPUT_FILE_PATH = sys.argv[1]
        print(f"Using user input file from command line: {USER_INPUT_FILE_PATH}")

    try:
        # 1. Create an instance of the Orchestrator
        pipeline = Orchestrator(
            config_path=CONFIG_FILE_PATH,
            user_input_path=USER_INPUT_FILE_PATH
        )

        # 2. Run the entire pipeline
        pipeline.run_pipeline()

    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: A required file was not found.")
        print(f"Details: {e}")
        print("Please ensure both 'config.json' and your user input file exist.")
    except Exception as e:
        print(f"\nAn unexpected critical error occurred: {e}")