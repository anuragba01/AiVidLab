from src.processors.script_generator import ScriptGenerator
import os
import traceback
from typing import List

try:
    from google import genai
except ImportError:
    raise ImportError("The 'google-generativeai' library is required. Please install it with 'pip install google-generativeai'")


# --- Independent Test Block ---
if __name__ == '__main__':
    print("\n--- Running Independent Test for ScriptGenerator ---")
    
    # These would come from a user or another program
    TEST_TOPICS = ["The philosophy of Stoicism", "Practical applications in modern life"]
    TEST_KEYWORDS = ["Marcus Aurelius", "emotional resilience", "mindfulness", "virtue"]
    TEST_TONE = "calm, inspirational, and educational"
    TEST_LENGTH = 250

    try:
        # 1. Instantiate the tool
        script_tool = ScriptGenerator(
            
            model_name="gemini-2.0-flash-lite" # Good model for this task
        )

        # 2. Use the tool to generate the script
        final_script = script_tool.process(
            topics=TEST_TOPICS,
            keywords=TEST_KEYWORDS,
            tone=TEST_TONE,
            target_word_count=TEST_LENGTH
        )

        # 3. Verify and print the output
        if final_script:
            print("\n--- SUCCESS: Script Generation Complete ---")
            
            # Save the script to a file to simulate the workflow
            output_filename = "test_generated_script.txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(final_script)
            
            print(f"  - Test script saved to '{output_filename}'")
            print("\n--- Generated Script ---")
            print(final_script)
            print("------------------------")
            # Check if the critical heading format was used
            if ":" in final_script and "::" in final_script:
                print("\nVerification PASSED: Script contains the required ':Heading::' format.")
            else:
                print("\nVerification FAILED: Script may be missing the required ':Heading::' format.")
        else:
            print("\nFAILURE: The processor returned None, indicating an error.")

    except (ValueError, RuntimeError, ImportError) as e:
        print(f"\nFAILURE: The test failed with an error: {e}")