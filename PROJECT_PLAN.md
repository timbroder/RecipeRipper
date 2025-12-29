# RecipeRipper - Technical Improvement Project Plan

**Version:** 1.0
**Date:** 2025-12-29
**Status:** Ready for Implementation

---

## Overview

This document outlines the technical improvements needed for the RecipeRipper project based on comprehensive QA analysis. The immediate blockers have been resolved, and this plan addresses the remaining issues categorized by priority.

## Completed (Sprint 0 - Immediate Fixes)

✅ **CRITICAL #1:** Reverted `compute_type` from `float16` to `int8` for CPU compatibility
✅ **BUG #1:** Added None check to `get_youtube_video_id()`
✅ **BUG #2:** Added None check to `combine_sources()`
✅ **BUG #3:** Added None check to `clean_directions()`
✅ **ISSUE #4:** Added try-finally block to `ocr_onscreen_text()` for proper resource cleanup
✅ **Reverted:** YouTube video format back to 1080p limit
✅ **Tests:** All 25 tests passing

---

## Sprint 1: Code Quality & Robustness (2-3 days)

**Goal:** Improve error handling, code quality, and add missing test coverage

### Tasks

#### 1.1: Improve Fallback Video Detection
**Priority:** HIGH
**Effort:** 2 hours
**Issue:** ISSUE #5 from QA report

**Current Problem:**
```python
# Only looks for .mp4, but format can download webm, mkv, etc.
mp4s = list(tmpdir.glob("*.mp4"))
```

**Implementation:**
```python
# Look for any video file format
for ext in ["mp4", "webm", "mkv", "flv", "avi"]:
    videos = list(tmpdir.glob(f"*.{ext}"))
    if videos:
        vid = videos[0]
        break
```

**Files:** `recipe_extractor.py:158-160`
**Tests:** Add `test_download_youtube_fallback_webm()`

---

#### 1.2: Remove Redundant cv2 Import
**Priority:** LOW
**Effort:** 10 minutes
**Issue:** ISSUE #6 from QA report

**Current Problem:**
```python
import cv2  # Line 201
# ...
import cv2 as _cv2  # Line 223 - REDUNDANT
```

**Implementation:**
- Remove line 223 (`import cv2 as _cv2`)
- Replace all `_cv2.` with `cv2.`

**Files:** `recipe_extractor.py:223-228`
**Tests:** Verify existing `test_ocr_onscreen_text` still passes

---

#### 1.3: Improve Exception Handling
**Priority:** MEDIUM
**Effort:** 3 hours
**Issue:** ISSUE #8 from QA report

**Current Problem:**
Multiple broad `except Exception:` blocks that hide errors (lines 126, 170, 178, 278, 388)

**Implementation:**
1. Add logging framework (use Python's `logging` module)
2. Replace broad exceptions with specific ones:
   ```python
   # Before
   except Exception:
       pass

   # After
   except (JSONDecodeError, IOError, FileNotFoundError) as e:
       logger.debug(f"Failed to parse info.json: {e}")
       return None
   ```

**Files:**
- `recipe_extractor.py` (multiple locations)
- Add logging configuration in `main()`

**Tests:** None required (internal improvement)

---

#### 1.4: Add Missing Test Coverage
**Priority:** HIGH
**Effort:** 4 hours

**Tests to Add:**

```python
# Null/None handling
def test_get_youtube_video_id_none_input():
    """Verify function handles None input gracefully"""
    assert get_youtube_video_id(None) is None
    assert get_youtube_video_id("") is None

def test_combine_sources_none_ocr_lines():
    """Verify function handles None ocr_lines"""
    result = combine_sources("desc", "trans", None)
    assert isinstance(result, tuple)
    assert len(result) == 2

def test_clean_directions_none_elements():
    """Verify function handles None elements in list"""
    result = clean_directions(["Step 1", None, "Step 2"])
    assert len(result) == 2
    assert "Step 1" in str(result)

# Resource management
def test_ocr_releases_on_exception(monkeypatch):
    """Verify VideoCapture.release() called even on exception"""
    released = []

    class DummyCapture:
        def isOpened(self): return True
        def get(self, prop): return 30.0 if prop == 5 else 3
        def grab(self): raise RuntimeError("Test exception")
        def release(self): released.append(True)

    monkeypatch.setitem(sys.modules, "cv2",
                       mock.Mock(VideoCapture=lambda p: DummyCapture()))

    with pytest.raises(RuntimeError):
        ocr_onscreen_text(Path("dummy.mp4"))

    assert released, "release() was not called"

# Edge cases
def test_transcribe_empty_video():
    """Test transcription with empty/corrupt video"""
    # Implementation

def test_ocr_zero_max_frames():
    """Test OCR with max_frames=0"""
    result = ocr_onscreen_text(Path("video.mp4"), max_frames=0)
    assert result == []

def test_ocr_negative_seconds_between():
    """Test OCR with negative seconds_between"""
    # Should handle gracefully with max(1, ...)
```

**Files:** `tests/test_recipe_extractor.py`
**Acceptance:** Increase test coverage from ~85% to >90%

---

## Sprint 2: Documentation & UX (1-2 days)

**Goal:** Update documentation to reflect changes and improve user experience

### Tasks

#### 2.1: Update README with New Defaults
**Priority:** HIGH
**Effort:** 1 hour

**Changes Needed:**

1. **Update resource requirements table:**
   ```markdown
   ## System Requirements

   | Model | Download | RAM | Speed | Accuracy | Recommended For |
   |-------|----------|-----|-------|----------|-----------------|
   | tiny | 75 MB | 0.5-1 GB | Very Fast | Basic | Quick tests |
   | small | 460 MB | 2-3 GB | Fast | Good | Most users |
   | **large-v3** | **3.1 GB** | **8-12 GB** | **Slow** | **Best** | **Default** |
   ```

2. **Add hardware recommendations:**
   ```markdown
   ### Hardware Recommendations
   - **Minimum:** 8 GB RAM, dual-core CPU
   - **Recommended:** 16 GB RAM, quad-core CPU
   - **For large-v3:** 16+ GB RAM recommended
   - **For slower systems:** Use `--model small` or `--model tiny`
   ```

3. **Update examples to show model override:**
   ```bash
   # Default (uses large-v3, requires 8-12 GB RAM)
   python recipe_extractor.py --youtube "URL"

   # For systems with limited RAM (recommended for <8 GB)
   python recipe_extractor.py --youtube "URL" --model small

   # For quick testing
   python recipe_extractor.py --youtube "URL" --model tiny
   ```

**Files:** `README.md`
**Acceptance:** README accurately reflects current defaults and provides migration guidance

---

#### 2.2: Add Migration Guide
**Priority:** MEDIUM
**Effort:** 30 minutes

**Implementation:**
Create `MIGRATION.md`:

```markdown
# Migration Guide

## Version 2.0 Changes

### Default Model Changed: small → large-v3

**What changed:**
- Default model increased from `small` (460 MB) to `large-v3` (3.1 GB)
- RAM requirements increased from 2-3 GB to 8-12 GB
- Processing time increased ~4x
- Accuracy significantly improved

**Who is affected:**
- Users with <8 GB RAM
- Automated/CI pipelines
- Docker containers with memory limits
- Scripts that don't specify `--model`

**How to maintain old behavior:**
Add `--model small` to all commands:
```bash
python recipe_extractor.py --youtube "URL" --model small
```

**Why the change:**
To provide the best accuracy by default for most users.
```

**Files:** Create `MIGRATION.md`
**Acceptance:** Users understand the breaking change

---

#### 2.3: Add Logging Configuration
**Priority:** MEDIUM
**Effort:** 2 hours

**Implementation:**

1. Add logging setup in `main()`:
   ```python
   import logging

   def main():
       parser = argparse.ArgumentParser(...)
       parser.add_argument("--verbose", "-v", action="store_true",
                          help="Enable verbose logging")
       parser.add_argument("--debug", action="store_true",
                          help="Enable debug logging")
       args = parser.parse_args()

       # Configure logging
       level = logging.DEBUG if args.debug else (
                logging.INFO if args.verbose else logging.WARNING)
       logging.basicConfig(
           level=level,
           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
       )
       logger = logging.getLogger(__name__)

       # Use logger throughout
       logger.info(f"Processing with model: {args.model}")
   ```

2. Replace `console.print` with `logger` in appropriate places

**Files:** `recipe_extractor.py`
**Tests:** Add `test_main_with_verbose_flag()`
**Acceptance:** Users can enable detailed logging for debugging

---

## Sprint 3: Advanced Features (3-5 days)

**Goal:** Add GPU support, performance improvements, and better UX

### Tasks

#### 3.1: GPU Auto-Detection
**Priority:** HIGH
**Effort:** 4 hours

**Implementation:**

```python
def get_device_and_compute_type():
    """Auto-detect best device and compute type for inference"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass

    # CPU fallback
    return "cpu", "int8"

def transcribe(video_path: Path, model_size: str="large-v3",
               language: Optional[str]=None,
               device: Optional[str]=None,
               compute_type: Optional[str]=None) -> str:
    from faster_whisper import WhisperModel

    if device is None or compute_type is None:
        auto_device, auto_compute = get_device_and_compute_type()
        device = device or auto_device
        compute_type = compute_type or auto_compute

    console.log(f"[cyan]Using device={device}, compute_type={compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    # ... rest
```

**CLI Changes:**
```python
parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                   help="Device for inference (auto-detect by default)")
parser.add_argument("--compute-type",
                   help="Compute type (auto-detect based on device by default)")
```

**Files:** `recipe_extractor.py`
**Tests:**
- `test_device_auto_detection_cpu()`
- `test_device_auto_detection_gpu()` (if CUDA available)

**Acceptance:** GPU automatically used when available, with 2-3x speedup

---

#### 3.2: Memory Check Before Model Loading
**Priority:** MEDIUM
**Effort:** 2 hours

**Implementation:**

```python
import psutil

MODEL_RAM_REQUIREMENTS = {
    "tiny": 1.0,    # GB
    "base": 2.0,
    "small": 3.0,
    "medium": 6.0,
    "large-v3": 12.0,
}

def check_memory_requirements(model_size: str) -> bool:
    """Check if system has enough RAM for the model"""
    required = MODEL_RAM_REQUIREMENTS.get(model_size, 12.0)
    available = psutil.virtual_memory().available / (1024**3)  # GB

    if available < required:
        console.print(f"[yellow]Warning: Model '{model_size}' requires ~{required} GB RAM")
        console.print(f"[yellow]Available: {available:.1f} GB")
        console.print(f"[yellow]Consider using a smaller model (--model small)")

        response = input("Continue anyway? (y/N): ")
        return response.lower() == 'y'

    return True

def main():
    # ... argument parsing ...

    if not check_memory_requirements(args.model):
        sys.exit(1)

    # ... continue processing ...
```

**Dependencies:** Add `psutil` to `requirements.txt`
**Files:** `recipe_extractor.py`
**Tests:** `test_memory_check_warning()`
**Acceptance:** Users warned before OOM errors occur

---

#### 3.3: Progress Indicators for Model Download
**Priority:** LOW
**Effort:** 3 hours

**Implementation:**

Use `rich.progress` for model download:

```python
from rich.progress import Progress, DownloadColumn, BarColumn, TextColumn, TimeRemainingColumn

def transcribe_with_progress(video_path: Path, ...):
    from faster_whisper import WhisperModel

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Loading {model_size} model...",
            total=MODEL_SIZES[model_size]
        )

        # This requires hooking into faster-whisper's download
        # May need custom implementation or PR to faster-whisper
```

**Note:** This may require changes to faster-whisper library or custom download logic

**Files:** `recipe_extractor.py`
**Acceptance:** Users see download progress, not just a hang

---

#### 3.4: Configuration File Support
**Priority:** LOW
**Effort:** 3 hours

**Implementation:**

Support `.reciperc` or `recipe.yaml` config file:

```yaml
# recipe.yaml
defaults:
  model: small
  language: en
  fps_sample: 0.5
  max_frames: 200
  cleanup: true
  outdir: ./output

# Override for specific use cases
profiles:
  quick:
    model: tiny
    max_frames: 50

  best:
    model: large-v3
    max_frames: 300
```

**CLI:**
```bash
# Use config file
python recipe_extractor.py --config recipe.yaml --youtube "URL"

# Use profile
python recipe_extractor.py --profile quick --youtube "URL"
```

**Dependencies:** Add `pyyaml` to `requirements.txt`
**Files:** Create `config.py`, modify `recipe_extractor.py`
**Tests:** `test_config_loading()`, `test_profile_override()`
**Acceptance:** Users can save preferences instead of typing flags

---

## Sprint 4: Code Improvements (Optional)

**Goal:** Long-term maintainability improvements

### Tasks

#### 4.1: Extract Magic Numbers to Constants
**Priority:** LOW
**Effort:** 1 hour

**Implementation:**

```python
# At top of file
DEFAULT_FPS_SAMPLE = 0.6
DEFAULT_MAX_FRAMES = 180
DEFAULT_OCR_LANGUAGE = "en"
DEFAULT_UPSCALE_THRESHOLD = 720
DEFAULT_UPSCALE_FACTOR = 1.25
MIN_OCR_TEXT_LENGTH = 2
MAX_DESCRIPTION_LENGTH = 5000
MAX_TRANSCRIPT_LENGTH = 10000
```

**Files:** `recipe_extractor.py`
**Acceptance:** No magic numbers in code

---

#### 4.2: Split into Multiple Modules
**Priority:** LOW
**Effort:** 4 hours

**Structure:**
```
recipeRipper/
  __init__.py
  __main__.py          # CLI entry point
  models.py            # Pydantic models
  youtube.py           # YouTube download logic
  transcription.py     # ASR logic
  ocr.py              # OCR logic
  parsing.py          # Ingredient/direction parsing
  cleaning.py         # Normalization logic
  config.py           # Configuration handling
  utils.py            # Utilities
```

**Files:** Refactor `recipe_extractor.py` into package
**Tests:** All existing tests must still pass
**Acceptance:** Better code organization, easier to navigate

---

## Sprint 5: Additional Test Coverage

**Goal:** Achieve >95% test coverage

### Tasks to Add

```python
# Integration tests
def test_full_pipeline_youtube_real()
def test_full_pipeline_local_real()

# Performance tests
def test_transcribe_performance_benchmark()
def test_ocr_performance_benchmark()

# Error cases
def test_corrupted_video_file()
def test_network_timeout_youtube()
def test_invalid_youtube_url()
def test_private_youtube_video()
def test_deleted_youtube_video()
def test_disk_full_during_cache()
def test_permission_denied_cache_dir()

# Model tests
def test_model_size_validation()
def test_invalid_model_name()
def test_all_supported_models()  # tiny, base, small, medium, large-v3

# OCR edge cases
def test_ocr_with_no_text_in_video()
def test_ocr_with_rotated_text()
def test_ocr_with_overlapping_text()

# Parsing edge cases
def test_parse_unicode_fractions()
def test_parse_metric_units()
def test_parse_imperial_units()
def test_parse_mixed_units()
def test_ingredient_without_quantity()
def test_direction_without_verb()
```

**Files:** `tests/test_recipe_extractor.py`, create `tests/test_integration.py`
**Effort:** 8 hours
**Acceptance:** Test coverage >95%

---

## Dependencies to Add

```txt
# requirements.txt additions
psutil>=5.9.0           # For memory checking (Sprint 3.2)
pyyaml>=6.0             # For config file support (Sprint 3.4)
```

---

## Success Metrics

### Sprint 1
- ✅ Test coverage increased to >90%
- ✅ All broad exceptions replaced with specific ones
- ✅ Logging framework in place
- ✅ Fallback video detection works for all formats

### Sprint 2
- ✅ README accurately reflects new defaults
- ✅ Migration guide published
- ✅ Users can enable debug logging

### Sprint 3
- ✅ GPU auto-detection working
- ✅ 2-3x speedup on GPU systems
- ✅ Memory warnings prevent OOM crashes
- ✅ Model download progress visible

### Sprint 4
- ✅ Codebase modularized
- ✅ No magic numbers
- ✅ Configuration file support

### Sprint 5
- ✅ Test coverage >95%
- ✅ Integration tests passing
- ✅ Performance benchmarks established

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU detection breaks CPU-only systems | Low | High | Extensive testing, fallback logic |
| Config file adds complexity | Medium | Low | Keep defaults simple, config optional |
| Refactoring breaks existing code | Low | High | Run full test suite after each change |
| Performance regression | Low | Medium | Add benchmarks, monitor performance |
| Users confused by new defaults | Medium | Medium | Clear documentation, migration guide |

---

## Timeline Estimate

- **Sprint 1 (Code Quality):** 2-3 days
- **Sprint 2 (Documentation):** 1-2 days
- **Sprint 3 (Advanced Features):** 3-5 days
- **Sprint 4 (Code Improvements):** 2-3 days (optional)
- **Sprint 5 (Test Coverage):** 2-3 days

**Total:** 10-16 days (2-3 weeks)

**Recommended Approach:** Execute Sprints 1-2 first (critical), then Sprint 3 (valuable), Sprints 4-5 as time permits.

---

## Notes

- All changes maintain backward compatibility where possible
- Breaking changes documented in MIGRATION.md
- Each sprint can be done independently
- Test coverage must not decrease
- All new features require tests
- Code review required before merging

---

**Project Plan Version:** 1.0
**Last Updated:** 2025-12-29
**Next Review:** After Sprint 1 completion
