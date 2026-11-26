# Python 3.14 and PaddleOCR

On-screen text OCR uses PaddleOCR, which depends on PaddlePaddle. As of November 2026, PaddlePaddle publishes no Python 3.14 (`cp314`) wheels: the newest builds on PyPI (3.3.1) and on Paddle's own stable and nightly package indexes stop at `cp313`. So `paddlepaddle`, and therefore `paddleocr`, can't be installed on Python 3.14 with pip.

## What happens on Python 3.14

- `requirements.txt` skips `paddleocr` and `paddlepaddle` on Python 3.14 (`python_version < "3.14"` markers), so `pip install -r requirements.txt` still succeeds.
- When a run reaches the OCR step, RecipeRipper prints a warning and skips it. The video description and speech transcript are still used for extraction, so recipes that are spoken or written in the description still come through. Text that only appears on screen is missed.
- `--preload-models` downloads the faster-whisper model and reports that the PaddleOCR preload failed.

## Recommended: use Python 3.13 for the full pipeline

Create the virtualenv with Python 3.13 (any of 3.10–3.13 works):

```bash
# macOS (Homebrew); pyenv or uv work too
brew install python@3.13
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have a 3.14 `.venv`, delete it and recreate it with the commands above.

## When PaddlePaddle ships Python 3.14 wheels

This command succeeds once a `cp314` wheel exists for your platform:

```bash
pip download paddlepaddle --no-deps --only-binary=:all: --python-version 3.14 -d /tmp/paddle-cp314-check
```

At that point, remove the `; python_version < "3.14"` markers from `requirements.txt`, remove the Python 3.14 notes from the README, and delete this file.
