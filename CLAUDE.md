# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RecipeRipper extracts structured ingredients and directions from recipe videos (YouTube or local files). It combines three text sources:
1. Video description (YouTube only)
2. Speech transcription via faster-whisper
3. On-screen text via PaddleOCR

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest

# Run single test
pytest tests/test_recipe_extractor.py::test_function_name -v

# Extract from YouTube (uses OpenAI by default, requires OPENAI_API_KEY)
python recipe_extractor.py --youtube "https://www.youtube.com/shorts/XXXXX" --cleanup

# Extract using local Ollama model (requires running Ollama)
python recipe_extractor.py --youtube "https://www.youtube.com/shorts/XXXXX" --cleanup --use-local

# Extract from local video
python recipe_extractor.py --video "/path/to/video.mp4" --cleanup

# Preload models for offline use
python recipe_extractor.py --preload-models --model small --language en
```

## Architecture

The codebase is a single-file Python script (`recipe_extractor.py`) with the following pipeline:

1. **Input**: YouTube URL or local video file
2. **Download** (YouTube only): Uses yt-dlp with local caching in `.cache/`
3. **Transcription**: faster-whisper ASR (CPU, int8 quantized)
4. **OCR**: PaddleOCR on sampled video frames
5. **Parsing**: LLM-based extraction (default) with heuristic fallback
   - LLM mode: Sends raw text to OpenAI (`--openai-model`, default `gpt-4o-mini`) or local Ollama (`--use-local`, `--local-model`)
   - Heuristic fallback: `looks_like_ingredient()` / `looks_like_direction()` regex-based classification
6. **Cleanup** (optional `--cleanup`): Normalizes units, merges duplicate ingredients, deduplicates and formats directions
7. **Output**: `output/recipe.json` and `output/recipe.md`

## Key Data Structures

- `Ingredient`: Pydantic model with `original`, `quantity`, `unit`, `item`, `notes` fields
- `RecipeOutput`: Contains `title`, `url`, `ingredients`, `directions`, `extras`, `raw_sources`

## Dependencies

Requires Python 3.10+ and ffmpeg. Main libraries: yt-dlp, faster-whisper, paddleocr, paddlepaddle, opencv-python, pydantic, rich, openai.
