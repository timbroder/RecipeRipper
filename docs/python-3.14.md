# Python 3.14 support for PaddleOCR

PaddlePaddle does not publish Python 3.14 wheels yet. PaddleOCR therefore is not installable directly via `pip install -r requirements.txt` on 3.14. Until upstream releases an official wheel, use a compiled wheel or build from source as outlined in [PaddlePaddle issue #71616](https://github.com/PaddlePaddle/Paddle/issues/71616).

## Quick CPU wheel install (recommended)
1. Download the community-provided `cp314` wheel linked in issue #71616. The attachment is usually named similar to `paddlepaddle-<version>-cp314-cp314-manylinux_x86_64.whl`.
2. Install the rest of the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the downloaded PaddlePaddle wheel:
   ```bash
   pip install /path/to/paddlepaddle-*-cp314-*.whl
   ```
4. Install PaddleOCR without re-pulling dependencies (so it reuses your custom PaddlePaddle install):
   ```bash
   pip install paddleocr==2.8.1 --no-deps
   ```

## Build PaddlePaddle from source (alternative)
If you prefer to build locally (for example, to match a specific CUDA/toolchain), follow the commands shared in issue #71616:
1. Clone the PaddlePaddle repository and check out the tag matching the version you want:
   ```bash
   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   git checkout release/2.6
   ```
2. Install Python build helpers (these are missing in some minimal Python installs and are required for `bdist_wheel`):
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```
3. Install the Python build requirements:
   ```bash
   pip install -r python/requirements.txt
   ```
4. Build the wheel (CPU example):
   ```bash
   python setup.py bdist_wheel
   ```
   The wheel is written to `./dist/`.
5. Install the freshly built wheel and PaddleOCR:
   ```bash
   pip install dist/paddlepaddle-*-cp314-*.whl
   pip install paddleocr==2.8.1 --no-deps
   ```

Either approach gives you a working PaddlePaddle installation on Python 3.14 so `recipe_extractor.py` can run normally.
