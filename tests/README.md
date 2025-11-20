
# Testing Workflow

This directory contains all automated tests for the project. The test suite uses `pytest` for both unit testing and end-to-end (E2E) testing.

## Prerequisites

Before running any tests, make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Environment Setup (Crucial)

Some tests require API access. To make the environment configuration simple and persistent, create a `.env` file in the project root and add:

```
GEMINI_API_KEY=your_actual_key_here
BYTEZ_API_KEY=your_actual_key_here
```

Both keys are required. Without them, integration and E2E tests may fail or be skipped.

## Running Tests

### 1. Unit Tests (Fast)

Run these to verify core logic without consuming API credits:

```bash
python -m pytest tests/unit_test/
```

### 2. End-to-End (E2E) Tests (Slow & Uses API Credits)
##### Disclaimer: e2e has not been tested yet, so it may cause unexperted behaviours.

These tests run the full workflow and generate real output such as audio/video files. Use the `-s` flag to show progress logs:

```bash
python -m pytest tests/e2e/ -s
```

### 3. Run All Tests

To execute the entire test suite (Unit + E2E):

```bash
python -m pytest
```

## Test Outputs

* E2E tests create an output directory such as `test_e2e_output/` (or `output/`) in the project root.
* These files are intentionally preserved so you can inspect generated audio/video.
* You may delete the folder manually when no longer needed.


