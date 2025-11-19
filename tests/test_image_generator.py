import os
from src.processors.image_generator import ImageGenerator
from dotenv import load_dotenv


# Independent Test Block 
if __name__ == '__main__':
    print("\n--- Running Independent Test for ImageGenerator ---")
    load_dotenv()
    
    test_api_key = os.getenv("GEMINI_API_KEY")
    
    if not test_api_key:
        print("\nWARNING: Please set the GEMINI_API_KEY environment variable for testing.")
        print("Skipping live API test.")
    else:
        TEST_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"
        TEST_PROMPT = "An ancient scroll unfurling on a dark wooden table, a single, elegant ink brush stroke in the sumi-e style is visible, soft, cinematic lighting from a single candle."
        TEST_NEGATIVE_PROMPT = "photo, realistic, 3d render, text, signature"

        try:
            image_tool = ImageGenerator(
                api_key=test_api_key,
                model_name=TEST_MODEL_NAME,
                bytez_api_key=os.getenv("BYTEZ_API_KEY")
            )

            generated_image_bytes = image_tool.process(
                prompt=TEST_PROMPT,
                negative_prompt=TEST_NEGATIVE_PROMPT
            )

            if generated_image_bytes:
                output_filename = "test_image_output.png"
                with open(output_filename, "wb") as f:
                    f.write(generated_image_bytes)
                print(f"\nSUCCESS: Test image file saved to '{output_filename}'")
            else:
                print("\nFAILURE: The processor returned None, indicating an error during generation.")

        except Exception as e:
            print(f"\nFAILURE: The test failed with an error: {e}")