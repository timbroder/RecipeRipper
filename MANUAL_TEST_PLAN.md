# RecipeRipper - Comprehensive Manual Test Plan

**Version:** 1.0
**Date:** 2026-01-14
**Environment:** Claude Code CLI
**Tester:** Claude (with human assistance for video URLs)

---

## Test Execution Notes

- ✅ = Test Passed
- ❌ = Test Failed
- ⚠️ = Test Passed with Issues
- 🔄 = Test In Progress
- ⏭️ = Test Skipped
- 📝 = Notes/Observations

---

## 1. Pre-Test Setup and Environment Validation

### 1.1 Dependency Verification
**Objective:** Ensure all system dependencies are installed and accessible

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| ENV-001 | Verify Python version (3.10+) | Python 3.10+ detected | | |
| ENV-002 | Verify FFmpeg installation | FFmpeg found in PATH | | |
| ENV-003 | Verify pip packages installed | All requirements.txt packages present | | |
| ENV-004 | Check .cache directory exists/writable | Directory exists or can be created | | |
| ENV-005 | Check output directory can be created | output/ dir can be created | | |

### 1.2 Repository State
**Objective:** Validate clean working state

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| ENV-006 | Check git repository status | Clean working directory | | |
| ENV-007 | Verify on correct branch | On claude/manual-test-plan-eW7nW | | |
| ENV-008 | Check for any uncommitted changes | No uncommitted changes | | |

---

## 2. Command-Line Interface Testing

### 2.1 Help and Information Commands
**Objective:** Verify all informational commands work correctly

| Test ID | Test Case | Command | Expected Result | Status | Notes |
|---------|-----------|---------|----------------|--------|-------|
| CLI-001 | Display help text | `python recipe_extractor.py --help` | Shows usage, all arguments | | |
| CLI-002 | List available models | `python recipe_extractor.py --list-models` | Shows model size table | | |
| CLI-003 | Run without arguments | `python recipe_extractor.py` | Shows error/usage | | |

### 2.2 Argument Validation
**Objective:** Verify input validation and error handling

| Test ID | Test Case | Command | Expected Result | Status | Notes |
|---------|-----------|---------|----------------|--------|-------|
| CLI-004 | Both --youtube and --video provided | `python recipe_extractor.py --youtube URL --video PATH` | Error: mutually exclusive | | |
| CLI-005 | Neither --youtube nor --video provided | `python recipe_extractor.py` | Error: one required | | |
| CLI-006 | Invalid --model value | `python recipe_extractor.py --youtube URL --model invalid` | Error: invalid model | | |
| CLI-007 | Invalid --fps-sample (negative) | `python recipe_extractor.py --youtube URL --fps-sample -1` | Error or reasonable default | | |
| CLI-008 | Invalid --max-frames (negative) | `python recipe_extractor.py --youtube URL --max-frames -1` | Error or reasonable default | | |
| CLI-009 | Non-existent --outdir parent | `python recipe_extractor.py --youtube URL --outdir /nonexistent/path/output` | Error or creates parent dirs | | |
| CLI-010 | Invalid language code | `python recipe_extractor.py --youtube URL --language invalid` | Error or fallback to auto | | |

---

## 3. YouTube Video Processing

### 3.1 URL Format Support
**Objective:** Test various YouTube URL formats are properly parsed

| Test ID | Test Case | URL Format | Expected Result | Status | Notes |
|---------|-----------|------------|----------------|--------|-------|
| YT-001 | Standard watch URL | `https://www.youtube.com/watch?v=VIDEO_ID` | Video ID extracted, download succeeds | | User provides URL |
| YT-002 | Shortened URL | `https://youtu.be/VIDEO_ID` | Video ID extracted, download succeeds | | User provides URL |
| YT-003 | Embed URL | `https://www.youtube.com/embed/VIDEO_ID` | Video ID extracted, download succeeds | | User provides URL |
| YT-004 | YouTube Shorts URL | `https://www.youtube.com/shorts/VIDEO_ID` | Video ID extracted, download succeeds | | User provides URL |
| YT-005 | Invalid YouTube URL | `https://youtube.com/invalid` | Error: cannot extract video ID | | |
| YT-006 | Non-YouTube URL | `https://vimeo.com/12345` | Error: not a YouTube URL | | |
| YT-007 | Private/unavailable video | URL to private video | Error: cannot download | | User provides URL if available |

### 3.2 Video Content Types
**Objective:** Test different types of recipe videos

| Test ID | Test Case | Video Type | Expected Result | Status | Notes |
|---------|-----------|------------|----------------|--------|-------|
| YT-008 | Short recipe video (< 1 min) | YouTube Short with recipe | Recipe extracted with ingredients & directions | | User provides URL |
| YT-009 | Medium recipe video (5-15 min) | Standard recipe tutorial | Recipe extracted with ingredients & directions | | User provides URL |
| YT-010 | Long recipe video (> 30 min) | Detailed cooking video | Recipe extracted with ingredients & directions | | User provides URL |
| YT-011 | Video with on-screen ingredient list | Recipe with text overlays | OCR captures on-screen ingredients | | User provides URL |
| YT-012 | Video with spoken ingredients only | Recipe spoken but not shown | Transcript captures ingredients | | User provides URL |
| YT-013 | Video with detailed description | Recipe in YouTube description | Description parsed for ingredients | | User provides URL |

### 3.3 Caching Behavior
**Objective:** Verify YouTube caching works correctly

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| YT-014 | First download of video | Video downloaded to .cache/, metadata saved | | |
| YT-015 | Re-run same video | Uses cached video, no re-download | | |
| YT-016 | Verify .cache/{video_id}.info.json | Metadata file exists and valid JSON | | |
| YT-017 | Verify .cache/{video_id}.{ext} | Video file exists and playable | | |
| YT-018 | Delete cache, re-run | Video re-downloaded | | |

---

## 4. Local Video File Processing

### 4.1 File Format Support
**Objective:** Test various video file formats

| Test ID | Test Case | File Format | Expected Result | Status | Notes |
|---------|-----------|-------------|----------------|--------|-------|
| VID-001 | MP4 file | `.mp4` video | Recipe extracted successfully | | User provides file |
| VID-002 | MKV file | `.mkv` video | Recipe extracted successfully | | User provides file |
| VID-003 | WebM file | `.webm` video | Recipe extracted successfully | | User provides file |
| VID-004 | MOV file | `.mov` video | Recipe extracted successfully | | User provides file |
| VID-005 | AVI file | `.avi` video | Recipe extracted successfully | | User provides file |

### 4.2 File Validation
**Objective:** Test error handling for invalid files

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| VID-006 | Non-existent file path | Error: file not found (exit code 2) | | |
| VID-007 | Invalid video file (corrupted) | Error: cannot process | | |
| VID-008 | Non-video file (e.g., .txt) | Error: invalid format | | |
| VID-009 | Empty file (0 bytes) | Error: invalid file | | |
| VID-010 | File path with spaces | Handles path correctly | | |
| VID-011 | Relative path | Resolves and processes correctly | | |
| VID-012 | Absolute path | Processes correctly | | |

---

## 5. Transcription Quality Testing

### 5.1 Model Size Comparison
**Objective:** Test different Whisper model sizes for accuracy and speed

| Test ID | Test Case | Model | Expected Result | Status | Notes |
|---------|-----------|-------|----------------|--------|-------|
| TRANS-001 | Process with tiny model | `--model tiny` | Fast, reasonable accuracy | | Time: ___ |
| TRANS-002 | Process with base model | `--model base` | Good balance | | Time: ___ |
| TRANS-003 | Process with small model (default) | `--model small` | Good accuracy | | Time: ___ |
| TRANS-004 | Process with medium model | `--model medium` | High accuracy | | Time: ___ |
| TRANS-005 | Process with large-v3 model | `--model large-v3` | Best accuracy, slow | | Time: ___ |
| TRANS-006 | Compare transcripts | All models on same video | Transcript quality comparison | | |

### 5.2 Language Support
**Objective:** Test transcription with different languages

| Test ID | Test Case | Language | Expected Result | Status | Notes |
|---------|-----------|----------|----------------|--------|-------|
| TRANS-007 | English video | `--language en` | Accurate transcription | | User provides URL |
| TRANS-008 | Spanish video | `--language es` | Accurate transcription | | User provides URL |
| TRANS-009 | French video | `--language fr` | Accurate transcription | | User provides URL |
| TRANS-010 | Auto-detect language | No language flag | Correct language detected | | User provides URL |
| TRANS-011 | Multi-language video | Mixed languages | Best-effort transcription | | User provides URL |

### 5.3 Audio Quality Scenarios
**Objective:** Test transcription with varying audio conditions

| Test ID | Test Case | Audio Condition | Expected Result | Status | Notes |
|---------|-----------|----------------|----------------|--------|-------|
| TRANS-012 | Clear audio, single speaker | High quality | Excellent transcription | | User provides URL |
| TRANS-013 | Background music/noise | Moderate noise | Reasonable transcription | | User provides URL |
| TRANS-014 | Multiple speakers | Overlapping voices | Best-effort transcription | | User provides URL |
| TRANS-015 | Accented speech | Strong accent | Reasonable transcription | | User provides URL |

---

## 6. OCR Text Extraction Testing

### 6.1 Frame Sampling Configuration
**Objective:** Test OCR with different sampling rates

| Test ID | Test Case | Configuration | Expected Result | Status | Notes |
|---------|-----------|---------------|----------------|--------|-------|
| OCR-001 | Default sampling | `--fps-sample 0.6` | Reasonable text capture | | |
| OCR-002 | Faster sampling | `--fps-sample 0.3` | More frames, slower | | |
| OCR-003 | Slower sampling | `--fps-sample 1.5` | Fewer frames, faster | | |
| OCR-004 | Max frames default | `--max-frames 180` | Up to 180 frames processed | | |
| OCR-005 | Low max frames | `--max-frames 50` | Faster, may miss text | | |
| OCR-006 | High max frames | `--max-frames 500` | Slower, more comprehensive | | |

### 6.2 On-Screen Text Scenarios
**Objective:** Verify OCR captures visible text

| Test ID | Test Case | Text Type | Expected Result | Status | Notes |
|---------|-----------|-----------|----------------|--------|-------|
| OCR-007 | Ingredient list overlay | On-screen bullet list | Ingredients captured in OCR | | User provides URL |
| OCR-008 | Recipe card shown | Static recipe card | All text extracted | | User provides URL |
| OCR-009 | Text with special characters | Fractions, degrees, etc. | Unicode captured correctly | | User provides URL |
| OCR-010 | Small text | Fine print | Text captured or missed gracefully | | User provides URL |
| OCR-011 | Handwritten text | Cursive/handwriting | Limited/no capture expected | | User provides URL |
| OCR-012 | Text on complex background | Low contrast | Best-effort capture | | User provides URL |

---

## 7. Recipe Parsing and Extraction

### 7.1 Ingredient Parsing
**Objective:** Test ingredient identification and parsing

| Test ID | Test Case | Input Type | Expected Result | Status | Notes |
|---------|-----------|------------|----------------|--------|-------|
| PARSE-001 | Simple ingredients | "1 cup flour" | Parsed: qty=1, unit=cup, item=flour | | |
| PARSE-002 | Fractional quantities | "1/2 tsp salt" | Parsed: qty=0.5, unit=tsp, item=salt | | |
| PARSE-003 | Mixed fractions | "1 1/2 cups sugar" | Parsed: qty=1.5, unit=cups, item=sugar | | |
| PARSE-004 | Unicode fractions | "½ cup milk" | Parsed: qty=0.5, unit=cup, item=milk | | |
| PARSE-005 | Ranges | "2-3 cloves garlic" | Parsed with range or midpoint | | |
| PARSE-006 | No unit | "3 eggs" | Parsed: qty=3, unit=None, item=eggs | | |
| PARSE-007 | Notes in parentheses | "1 cup flour (sifted)" | Parsed with notes="sifted" | | |
| PARSE-008 | Multiple units | "1 lb (450g) butter" | Parsed with primary unit | | |
| PARSE-009 | Bullet point ingredients | "- 1 tsp vanilla" | Bullet removed, ingredient parsed | | |
| PARSE-010 | Numbered ingredients | "1. 2 cups flour" | Number prefix handled | | |

### 7.2 Direction Parsing
**Objective:** Test cooking direction identification

| Test ID | Test Case | Input Type | Expected Result | Status | Notes |
|---------|-----------|------------|----------------|--------|-------|
| PARSE-011 | Imperative verb start | "Mix flour and eggs" | Identified as direction | | |
| PARSE-012 | Temperature reference | "Bake at 350°F" | Identified as direction | | |
| PARSE-013 | Time reference | "Cook for 20 minutes" | Identified as direction | | |
| PARSE-014 | Numbered steps | "1. Preheat oven" | Number removed, direction extracted | | |
| PARSE-015 | Multi-sentence direction | "Mix well. Set aside." | Both sentences in direction | | |

### 7.3 Source Combination
**Objective:** Test combining description, transcript, and OCR

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| PARSE-016 | All sources present | Description + transcript + OCR combined | | |
| PARSE-017 | Description only | Uses available description | | |
| PARSE-018 | Transcript only | Uses available transcript | | |
| PARSE-019 | OCR only | Uses available OCR text | | |
| PARSE-020 | Overlapping content | Deduplication works correctly | | |

---

## 8. Cleanup Mode Testing

### 8.1 Ingredient Cleanup
**Objective:** Test ingredient normalization with --cleanup flag

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| CLEAN-001 | Unit normalization | "tablespoon" → "tbsp" | | |
| CLEAN-002 | Unicode fraction conversion | "½" → "1/2" | | |
| CLEAN-003 | Duplicate ingredients | Same ingredient merged with added quantities | | |
| CLEAN-004 | Case normalization | Consistent casing applied | | |
| CLEAN-005 | Unit plural handling | "cups" vs "cup" normalized | | |

### 8.2 Direction Cleanup
**Objective:** Test direction normalization

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| CLEAN-006 | Timestamp removal | "[00:30]" removed from directions | | |
| CLEAN-007 | Abbreviation expansion | "min" → "minute", "hr" → "hour" | | |
| CLEAN-008 | Case fixing | Proper sentence case applied | | |
| CLEAN-009 | Duplicate removal | Identical directions removed | | |
| CLEAN-010 | Fragment merging | Short fragments merged into complete steps | | |

### 8.3 Cleanup vs No-Cleanup Comparison
**Objective:** Compare output with and without cleanup

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| CLEAN-011 | Same video, no --cleanup | Raw output with minimal processing | | |
| CLEAN-012 | Same video, with --cleanup | Normalized, deduplicated output | | |
| CLEAN-013 | Quality comparison | Cleanup improves readability | | |

---

## 9. Output File Generation

### 9.1 JSON Output Validation
**Objective:** Verify recipe.json structure and content

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| OUT-001 | JSON file created | output/recipe.json exists | | |
| OUT-002 | Valid JSON syntax | File parses as valid JSON | | |
| OUT-003 | Required fields present | title, ingredients, directions present | | |
| OUT-004 | Ingredient structure | Each ingredient has required fields | | |
| OUT-005 | URL field (YouTube) | URL present for YouTube videos | | |
| OUT-006 | URL field (local) | URL is null for local files | | |
| OUT-007 | extras.ocr_samples | First 50 OCR samples included | | |
| OUT-008 | raw_sources | Description and transcript truncated | | |

### 9.2 Markdown Output Validation
**Objective:** Verify recipe.md formatting and content

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| OUT-009 | Markdown file created | output/recipe.md exists | | |
| OUT-010 | Proper markdown syntax | Valid markdown formatting | | |
| OUT-011 | Title and source | Header with title and source link | | |
| OUT-012 | Ingredients section | "## Ingredients" with bulleted list | | |
| OUT-013 | Directions section | "## Directions" with numbered list | | |
| OUT-014 | Human readable | Clean, readable format | | |

### 9.3 Output Directory Configuration
**Objective:** Test custom output directories

| Test ID | Test Case | Command | Expected Result | Status | Notes |
|---------|-----------|---------|----------------|--------|-------|
| OUT-015 | Default output dir | No --outdir flag | Files in ./output/ | | |
| OUT-016 | Custom output dir | `--outdir custom_output` | Files in ./custom_output/ | | |
| OUT-017 | Nested output dir | `--outdir results/test1` | Directories created as needed | | |
| OUT-018 | Absolute path output | `--outdir /tmp/recipes` | Files in specified absolute path | | |

---

## 10. Model Preloading and Offline Mode

### 10.1 Model Download
**Objective:** Test model preloading functionality

| Test ID | Test Case | Command | Expected Result | Status | Notes |
|---------|-----------|---------|----------------|--------|-------|
| PRE-001 | Preload default model | `--preload-models` | small model downloaded | | |
| PRE-002 | Preload specific model | `--preload-models --model tiny` | tiny model downloaded | | |
| PRE-003 | Preload with language | `--preload-models --language en` | Model + language data cached | | |
| PRE-004 | Verify model cache | Check HF cache dir | Models present in cache | | |
| PRE-005 | Verify OCR models | PaddleOCR models | OCR models downloaded | | |

### 10.2 Offline Operation
**Objective:** Verify operation without internet (simulated)

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| PRE-006 | Process with cached models | Uses local models, no download | | Requires cached models |
| PRE-007 | Process new video offline | Downloads fail gracefully | | Disconnect if possible |

---

## 11. Error Handling and Edge Cases

### 11.1 System Errors
**Objective:** Test graceful handling of system issues

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| ERR-001 | FFmpeg not installed | Error message, exit code 1 | | May need to uninstall FFmpeg |
| ERR-002 | Insufficient disk space | Error or warning | | Hard to simulate |
| ERR-003 | Permission denied (output dir) | Error: cannot write | | |
| ERR-004 | Network timeout (YouTube) | Error: download failed | | |

### 11.2 Content Edge Cases
**Objective:** Test unusual or problematic content

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| ERR-005 | Video with no audio | OCR only, graceful handling | | User provides URL |
| ERR-006 | Video with no visual text | Transcript only | | User provides URL |
| ERR-007 | Non-recipe video | Extracts what it can, may be nonsense | | User provides URL |
| ERR-008 | Very short video (< 10s) | Processes successfully or errors gracefully | | User provides URL |
| ERR-009 | Very long video (> 1hr) | Processes successfully or hits timeout | | User provides URL |
| ERR-010 | Silent video | Graceful handling, no transcript | | User provides URL |

### 11.3 Subprocess Timeout
**Objective:** Test timeout handling

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| ERR-011 | Process exceeds timeout | Process killed, error reported | | Hard to simulate |

---

## 12. Performance and Resource Usage

### 12.1 Performance Benchmarks
**Objective:** Record performance metrics for different configurations

| Test ID | Test Case | Configuration | Expected Result | Status | Time | Memory |
|---------|-----------|---------------|----------------|--------|------|--------|
| PERF-001 | Small video, tiny model | < 1 min video, --model tiny | Fast processing | | | |
| PERF-002 | Small video, large model | < 1 min video, --model large-v3 | Slower but accurate | | | |
| PERF-003 | Medium video, default | 5-10 min video, --model small | Reasonable time | | | |
| PERF-004 | Minimal OCR | --max-frames 20 | Fast OCR phase | | | |
| PERF-005 | Extensive OCR | --max-frames 500 | Slow OCR phase | | | |

### 12.2 Resource Monitoring
**Objective:** Monitor system resource usage

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| PERF-006 | CPU usage | Reasonable CPU usage during transcription | | |
| PERF-007 | Memory usage | Within expected limits per model size | | |
| PERF-008 | Disk usage | Cache files cleaned up appropriately | | |

---

## 13. Integration and End-to-End Tests

### 13.1 Complete Workflows
**Objective:** Test full user workflows

| Test ID | Test Case | Workflow | Expected Result | Status | Notes |
|---------|-----------|----------|----------------|--------|-------|
| E2E-001 | First-time user: YouTube | Download, process, view output | Success, files created | | |
| E2E-002 | First-time user: local file | Process, view output | Success, files created | | |
| E2E-003 | Re-process same video | Use cache, fast processing | Success, uses cache | | |
| E2E-004 | Process multiple videos | Sequential processing | All succeed | | |
| E2E-005 | Full cleanup workflow | Process with --cleanup | Clean, normalized output | | |

### 13.2 Real-World Recipe Scenarios
**Objective:** Test with diverse real recipe videos

| Test ID | Test Case | Recipe Type | Expected Result | Status | Notes |
|---------|-----------|-------------|----------------|--------|-------|
| E2E-006 | Baking recipe | Precise measurements | Accurate ingredient extraction | | User provides URL |
| E2E-007 | Cooking recipe | Flexible measurements | Reasonable extraction | | User provides URL |
| E2E-008 | Quick recipe (Short) | 30-60 second video | Complete extraction | | User provides URL |
| E2E-009 | Complex multi-part recipe | Multiple components | All parts captured | | User provides URL |
| E2E-010 | International recipe | Non-English | Language-appropriate extraction | | User provides URL |

---

## 14. User Experience and Usability

### 14.1 Output Quality Assessment
**Objective:** Evaluate human-readable quality

| Test ID | Test Case | Assessment Criteria | Expected Result | Status | Notes |
|---------|-----------|---------------------|----------------|--------|-------|
| UX-001 | Ingredient completeness | All ingredients captured | No missing ingredients | | Human review |
| UX-002 | Ingredient accuracy | Quantities correct | Measurements accurate | | Human review |
| UX-003 | Direction completeness | All steps present | Complete instructions | | Human review |
| UX-004 | Direction ordering | Logical order | Steps in correct sequence | | Human review |
| UX-005 | Markdown readability | Easy to read | Clear, well-formatted | | Human review |

### 14.2 Error Messages and Feedback
**Objective:** Evaluate user-facing messages

| Test ID | Test Case | Expected Result | Status | Notes |
|---------|-----------|----------------|--------|-------|
| UX-006 | Clear error messages | Errors are understandable | | |
| UX-007 | Progress indicators | User knows processing status | | |
| UX-008 | Success confirmation | Clear completion message | | |
| UX-009 | Helpful validation errors | Suggests fixes for invalid input | | |

---

## 15. Regression Testing

### 15.1 Known Issues
**Objective:** Verify previously fixed issues remain fixed

| Test ID | Test Case | Previous Issue | Expected Result | Status | Notes |
|---------|-----------|----------------|----------------|--------|-------|
| REG-001 | (To be filled during testing) | | | | |

---

## Test Summary

### Coverage Statistics
- **Total Test Cases:** ~170
- **Setup/Environment:** 8 tests
- **CLI Testing:** 13 tests
- **YouTube Processing:** 18 tests
- **Local Video Processing:** 12 tests
- **Transcription:** 15 tests
- **OCR:** 12 tests
- **Parsing:** 20 tests
- **Cleanup Mode:** 13 tests
- **Output Files:** 14 tests
- **Model Preloading:** 7 tests
- **Error Handling:** 11 tests
- **Performance:** 8 tests
- **End-to-End:** 10 tests
- **User Experience:** 9 tests

### Test Execution Timeline
- **Estimated Duration:** 6-10 hours (with model downloads and video processing)
- **Priority:** High-priority tests marked in execution order

### Critical Path Tests (Must Pass)
1. ENV-001 through ENV-005: Environment setup
2. CLI-001: Basic help functionality
3. YT-001, YT-004: YouTube URL support (standard + Shorts)
4. VID-001: Local MP4 support
5. TRANS-003: Default model transcription
6. OCR-001: Default OCR sampling
7. PARSE-001 through PARSE-015: Core parsing logic
8. OUT-001 through OUT-014: Output file generation
9. E2E-001, E2E-002: Complete workflows

### Test Environment Requirements
1. Internet connection (for YouTube tests, model downloads)
2. Sample video files (MP4, MKV, WebM, MOV, AVI)
3. Sample YouTube URLs:
   - Regular recipe video
   - YouTube Short recipe
   - Video with on-screen text
   - Video with spoken-only recipe
   - Video with detailed description
4. At least 10GB free disk space (for models and cache)
5. Python 3.10+ environment with all dependencies

---

## Test Execution Strategy

### Phase 1: Setup and Validation (30 minutes)
- Run all ENV tests
- Run all CLI information tests
- Verify basic functionality

### Phase 2: Core Functionality (2-3 hours)
- YouTube processing (YT-001 through YT-013)
- Local video processing (VID-001 through VID-006)
- Basic transcription and OCR (TRANS-001, OCR-001)

### Phase 3: Feature Deep Dive (2-3 hours)
- All transcription model tests
- All OCR configuration tests
- All parsing tests
- Output file validation

### Phase 4: Advanced Features (1-2 hours)
- Cleanup mode testing
- Model preloading
- Caching verification
- Custom configurations

### Phase 5: Edge Cases and Errors (1-2 hours)
- All error handling tests
- Edge case scenarios
- Performance benchmarks

### Phase 6: End-to-End and UX (1-2 hours)
- Complete workflow tests
- Real-world recipe scenarios
- User experience evaluation

---

## Bug Reporting Template

**Bug ID:** BUG-XXX
**Test ID:** (Related test case)
**Severity:** Critical / High / Medium / Low
**Description:**
**Steps to Reproduce:**
1. Step 1
2. Step 2
3. ...

**Expected Result:**
**Actual Result:**
**Environment:**
**Additional Notes:**

---

## Test Sign-Off

**Tester:** Claude (AI Assistant)
**Reviewer:** (Human reviewer)
**Start Date:** 2026-01-14
**Completion Date:** ___________
**Overall Result:** Pass / Fail / Partial
**Recommendation:** Ready for release / Needs fixes / Requires retesting

---

## Appendix A: Sample Test Data

### Sample YouTube URLs Needed
1. Standard recipe video (5-15 min)
2. YouTube Short recipe (< 1 min)
3. Recipe with on-screen ingredient list
4. Recipe with spoken-only content
5. Recipe in Spanish/French (if available)
6. Baking recipe (precise measurements)
7. Cooking recipe (flexible measurements)
8. Silent video (if needed for edge case)

### Sample Local Video Files Needed
1. test_recipe.mp4
2. test_recipe.mkv
3. test_recipe.webm
4. test_recipe.mov
5. test_recipe.avi
6. invalid_file.txt (for error testing)

---

## Appendix B: Success Criteria

### Minimum Passing Criteria
- All critical path tests pass
- No critical or high-severity bugs
- Core functionality (YouTube, local files, transcription, OCR, parsing) works reliably
- Output files generated correctly
- Basic error handling works

### Ideal Passing Criteria
- 95%+ of all tests pass
- No medium-severity bugs
- All edge cases handled gracefully
- Excellent performance across configurations
- High-quality, human-readable output
- Clear error messages and user feedback

---

*End of Manual Test Plan*
