# Recipe Video Extractor (YouTube or Local File)

[![Tests](https://github.com/RecipeRipper/RecipeRipper/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/RecipeRipper/RecipeRipper/actions/workflows/tests.yml)

Extract structured **ingredients** and **directions** from a recipe video — works with a **YouTube URL** (incl. Shorts) or a **local file**.

## What it does
- Downloads video + description JSON (if YouTube) using `yt-dlp`
- Transcribes speech locally with `faster-whisper` (no API keys)
- OCRs on-screen text from sampled frames via `PaddleOCR`
- Merges description + transcript + on-screen text
- Heuristically splits **Ingredients** vs **Directions**
- Outputs:
  - `output/recipe.json` (structured)
  - `output/recipe.md` (human-readable)

## Install (macOS/Linux)
> You need Python 3.10+ and `ffmpeg`.
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# If you don't have ffmpeg:
#   macOS (brew): brew install ffmpeg
#   Ubuntu/Debian: sudo apt-get install -y ffmpeg
```

## Usage
```bash
# From this folder:
python recipe_extractor.py --youtube "https://www.youtube.com/shorts/XXXXX"

# OR local file:
python recipe_extractor.py --video "/path/to/video.mp4"

# Options
#   --language en            # whisper language hint (auto by default)
#   --model small            # faster-whisper model size: tiny/base/small/medium/large-v3
#   --fps-sample 0.5         # seconds between OCR frames (default 0.6s)
#   --max-frames 180         # OCR at most N frames
#   --outdir output          # where JSON/MD goes
#   --cleanup                # normalize units, fractions, dedupe ingredients, tidy steps
```

### 🧪 Quick Test Example
```bash
# Example 1 — YouTube Short
python recipe_extractor.py   --youtube "https://www.youtube.com/shorts/5fEwO2kS64A"   --model small   --cleanup   --outdir output

# Example 2 — Local video
python recipe_extractor.py   --video "/Users/you/Videos/pancake_recipe.mp4"   --model small   --fps-sample 0.5   --max-frames 150   --cleanup   --outdir output
```

## Cleanup mode
Add `--cleanup` to normalize ingredients (units, fractions, dedupe) and tidy directions (remove timestamps, fix casing, merge fragments).

Example:
```bash
python recipe_extractor.py --youtube "https://www.youtube.com/shorts/XXXXX" --cleanup
```

## Running tests
The project uses `pytest` for its unit and integration tests.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

### Continuous integration

GitHub Actions automatically runs the test suite for every pull request and for pushes to the `main` branch via the `Tests` workflow in `.github/workflows/tests.yml`. Mark the "Tests" status check as required in your branch protection rules to prevent merges when the suite fails.


## Offline prep (optional)
To cache all models ahead of time (useful for travel/air-gapped machines), run:
```bash
python recipe_extractor.py --preload-models --model small --language en
```
This downloads the faster‑whisper ASR model you specify (e.g., `small`) and the PaddleOCR English models to your local cache.


## List available ASR model sizes
See recommended Faster‑Whisper sizes and rough resource needs:
```bash
python recipe_extractor.py --list-models
```
