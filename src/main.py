import os
import sys
from dotenv import load_dotenv
from .orchestrator import Orchestrator

# Load environment variables from the .env file
load_dotenv()

# Define the paths to your configuration and input files.
CONFIG_FILE_PATH = "config.json"
INPUT_FILE_PATH = "input.json"

if __name__ == "__main__":
    
    # Get the API key securely from the environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        print("\nCRITICAL ERROR: 'GEMINI_API_KEY' not found in the environment.")
        print("Please create a .env file and add the line: GEMINI_API_KEY='your_key_here'")
        sys.exit(1) # Exit the script with an error code

    # Allow passing a different input file via command line
    if len(sys.argv) > 1:
        INPUT_FILE_PATH = sys.argv[1]
        print(f"Using input file from command line: {INPUT_FILE_PATH}")

    try:
        pipeline = Orchestrator(
            config_path=CONFIG_FILE_PATH,
            input_path=INPUT_FILE_PATH,
        )
        pipeline.run_pipeline()

    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: A required file was not found: {e}")
        print(f"Please ensure '{CONFIG_FILE_PATH}' and '{INPUT_FILE_PATH}' both exist.")
    except Exception as e:
        print(f"\nAn unexpected critical error occurred: {e}")