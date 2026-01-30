# RecipeRipper

[![Tests](https://github.com/timbroder/RecipeRipper/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/timbroder/RecipeRipper/actions/workflows/tests.yml)

Extract structured **ingredients** and **directions** from a recipe video — works with a **YouTube URL** (incl. Shorts) or a **local file**.

## How it works

1. **Download** the video and description (YouTube) or read from disk (local file)
2. **Description-first check** — when LLM extraction is active, the description is sent to the LLM first. If it contains a complete recipe (at least 2 ingredients and 1 direction), transcription and OCR are skipped entirely.
3. **Transcribe** speech with `faster-whisper` (local, no API keys)
4. **OCR** on-screen text from sampled frames via `PaddleOCR`
5. **Extract** ingredients and directions using an LLM (OpenAI or Ollama), with a heuristic fallback
6. **Cleanup** (optional) — normalize units, deduplicate ingredients, tidy directions
7. **Output** `recipe.json` (structured) and `recipe.md` (human-readable)

## Requirements

- Python 3.10+
- `ffmpeg`
- **OpenAI API key** (for default LLM extraction) _or_ a running **Ollama** instance (for local extraction)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# If you don't have ffmpeg:
#   macOS:          brew install ffmpeg
#   Ubuntu/Debian:  sudo apt-get install -y ffmpeg

# Set your OpenAI key (required for default LLM extraction):
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
# YouTube (uses OpenAI by default)
python recipe_extractor.py --youtube "https://www.youtube.com/shorts/XXXXX" --cleanup

# Local video
python recipe_extractor.py --video "/path/to/video.mp4" --cleanup

# Use a local Ollama model instead of OpenAI
python recipe_extractor.py --youtube "https://www.youtube.com/shorts/XXXXX" --cleanup --use-local

# Verbose output — logs each pipeline step
python recipe_extractor.py --youtube "https://www.youtube.com/shorts/XXXXX" --cleanup --verbose

# Publish output to a public GitHub Gist (requires gh CLI)
python recipe_extractor.py --youtube "https://www.youtube.com/shorts/XXXXX" --cleanup --publish
```

### CLI reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--youtube` | string | — | YouTube URL (works with Shorts). Mutually exclusive with `--video`. |
| `--video` | string | — | Path to a local video file. Mutually exclusive with `--youtube`. |
| `--language` | string | auto-detect | Language hint for transcription (e.g. `en`, `es`, `fr`) |
| `--model` | string | `small` | faster-whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--fps-sample` | float | `0.6` | Seconds between OCR frames |
| `--max-frames` | int | `180` | Maximum number of frames to OCR |
| `--outdir` | string | `output` | Directory for `recipe.json` and `recipe.md` |
| `--cleanup` | flag | off | Normalize units, deduplicate ingredients, tidy directions |
| `--use-local` | flag | off | Use a local Ollama model instead of OpenAI |
| `--local-model` | string | `llama3.1:8b-instruct` | Ollama model name |
| `--openai-model` | string | `gpt-4o-mini` | OpenAI model name |
| `--preload-models` | flag | off | Download/cache ASR and OCR models for offline use |
| `--list-models` | flag | off | Show recommended faster-whisper sizes and resource needs |
| `--verbose` | flag | off | Log each pipeline step to the console |
| `--publish` | flag | off | Upload output to a public GitHub Gist (requires `gh` CLI) |

## LLM extraction

By default, raw text (description + transcript + OCR) is sent to an LLM which returns structured ingredients and directions. If the LLM is unavailable or the response can't be parsed, extraction falls back to heuristic regex-based classification.

- **OpenAI (default)** — requires `OPENAI_API_KEY`. Model can be changed with `--openai-model`.
- **Ollama (local)** — pass `--use-local`. Requires a running Ollama instance. Model can be changed with `--local-model`.
- **Description-first optimization** — when LLM extraction is active and a YouTube video has a text description, the LLM tries extracting from the description alone first. If it finds a complete recipe (>= 2 ingredients, >= 1 direction), the expensive transcription and OCR steps are skipped.

## Cleanup mode

Add `--cleanup` to post-process the extracted recipe:
- Normalize units and fractions in ingredients
- Merge duplicate ingredients
- Remove timestamps, fix casing, and deduplicate directions

## Offline prep

Cache the ASR and OCR models ahead of time (useful for air-gapped machines):

```bash
python recipe_extractor.py --preload-models --model small --language en
```

List available faster-whisper model sizes:

```bash
python recipe_extractor.py --list-models
```

## Output

Results are saved to the `--outdir` directory (default `output/`):

- **`recipe.json`** — structured JSON with `title`, `url`, `ingredients`, `directions`, `extras`, and `raw_sources`
- **`recipe.md`** — human-readable Markdown

## Running tests

```bash
pytest
```

GitHub Actions runs the test suite on every pull request and push to `main`. See the Tests badge above.
