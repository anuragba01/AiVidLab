from src.processors.subtitle_processor import SubtitleProcessor

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("Running standalone SubtitleProcessor test…")

    # Minimal fake word-timestamp data (NO audio needed)
    # Simulates a tiny 2-second speech segment.

    word_timestamps = [
        {"text": "Hello", "start": 0.00, "end": 0.50},
        {"text": "world", "start": 0.60, "end": 1.00},
        {"text": "this",  "start": 1.20, "end": 1.40},
        {"text": "works", "start": 1.50, "end": 1.90},
    ]

    # Example headings to align
    heading_strings = [
        "Hello world"
    ]

    # Minimal config (safe defaults)
    style_config = {
        "default": {
            "font_name": "Arial",
            "font_size": 72,
            "primary_colour": "&H00FFFFFF",
            "outline_colour": "&H00000000",
            "back_colour": "&H99000000",
            "alignment": 2
        },
        "line_rules": {
            "max_words_per_line": 7,
            "max_line_duration_s": 5,
            "gap_threshold_s": 0.4
        }
    }

    heading_style_config = {
        "heading": {
            "font_name": "Arial",
            "font_size": 86,
            "primary_colour": "&H00FFFF00",
            "outline_colour": "&H00000000",
            "back_colour": "&H60000000",
            "alignment": 5
        }
    }

    processor = SubtitleProcessor()

    logger.info("Testing SubtitleProcessor with fake timestamps…")

    try:
        ass_content = processor.process(
            style_config=style_config,
            heading_style_config=heading_style_config,
            video_width=1920,
            video_height=1080,
            word_timestamps=word_timestamps,
            heading_strings=heading_strings,
            time_offset_s=0.0
        )

        logger.info("Test SUCCESS — Subtitle content generated.")
        print("\n--- Generated ASS Subtitle Content ---\n")
        print(ass_content)
        print("\n--------------------------------------\n")

    except Exception as e:
        logger.error(f"Test FAILED — {e}")
