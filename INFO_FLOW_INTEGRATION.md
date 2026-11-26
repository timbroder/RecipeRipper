# INFO_FLOW_INTEGRATION.md — Integration & Infrastructure

How the script manages configuration, caching, model lifecycle, and publishing. This document covers the parts a Flutter app would need to replicate beyond the core data pipeline.

---

## CLI arguments and configuration

The script is configured entirely via command-line arguments. A Flutter app would translate these into settings/preferences.

### Input source (mutually exclusive, one required)

| Argument | Type | Purpose |
|----------|------|---------|
| `--youtube URL` | string | YouTube URL to process (Shorts, watch, embed all supported) |
| `--video PATH` | string | Local video file path |

### Transcription settings

| Argument | Default | Purpose |
|----------|---------|---------|
| `--language` | auto-detect | 2-letter language code (e.g. `en`, `es`, `fr`). Improves accuracy when known |
| `--model` | `small` | Whisper model size. Options: `tiny`, `base`, `small`, `medium`, `large-v3` |

**Model size trade-offs:**

| Size | Download | RAM | Notes |
|------|----------|-----|-------|
| `tiny` | ~75 MB | ~0.5-1 GB | Fastest, lower accuracy |
| `base` | ~145 MB | ~1-2 GB | Balanced for Shorts |
| `small` | ~460 MB | ~2-3 GB | Recommended default |
| `medium` | ~1.5 GB | ~4-6 GB | Better accuracy |
| `large-v3` | ~3.1 GB | ~8-12 GB | Best accuracy, slowest |

### OCR settings

| Argument | Default | Purpose |
|----------|---------|---------|
| `--fps-sample` | `0.6` | Seconds between sampled frames |
| `--max-frames` | `180` | Maximum frames to OCR |

At the default 0.6s interval, a 60-second video produces ~100 frames. The `max_frames` cap prevents runaway processing on long videos.

### LLM settings

| Argument | Default | Purpose |
|----------|---------|---------|
| `--use-local` | false | Use local Ollama instead of OpenAI |
| `--local-model` | `qwen3:8b` | Ollama model name |
| `--openai-model` | `gpt-4o-mini` | OpenAI model name |

When neither `--use-local` is set nor `OPENAI_API_KEY` is available, the pipeline falls back to heuristic extraction (no LLM).

### Output settings

| Argument | Default | Purpose |
|----------|---------|---------|
| `--outdir` | `output` | Directory for JSON and Markdown files |
| `--cleanup` | false | Normalize units, merge duplicate ingredients, tidy directions |
| `--publish` | false | Upload Markdown to a public GitHub Gist |
| `--verbose` | false | Print detailed progress logs |

### Model management

| Argument | Purpose |
|----------|---------|
| `--preload-models` | Download Whisper and PaddleOCR models for offline use |
| `--list-models` | Show available Whisper model sizes with resource requirements |

---

## Caching strategy

### YouTube video cache

**Location:** `.cache/` directory relative to the script

**Structure:**
```
.cache/
  {video_id}.mp4          # Downloaded video
  {video_id}.info.json    # yt-dlp metadata (title, description, etc.)
```

**Cache check logic:**
1. Extract the 11-character video ID from the URL
2. Glob for `.cache/{vid_id}.*` matching extensions `.mp4`, `.mkv`, `.webm`
3. If found, load the video and parse the `.info.json` — skip the download entirely
4. If not found, download via yt-dlp and save both files

**Cache invalidation:** None. Files are kept indefinitely. To re-download, the user deletes files from `.cache/` manually.

**For Flutter:** This maps naturally to an app cache directory. Key decisions:
- Cache by video ID for deduplication
- Store metadata alongside the video for offline access to title/description
- Consider cache size limits and expiration policies

### Whisper model cache

faster-whisper models are cached by the library itself (typically in `~/.cache/huggingface/`). The `--preload-models` flag triggers an explicit download ahead of time.

### PaddleOCR model cache

PaddleOCR models are cached in `~/.paddlex/official_models/`. The `--preload-models` flag triggers download. Models include:
- `PP-LCNet_x1_0_doc_ori` — document orientation
- `UVDoc` — document unwarping
- `PP-LCNet_x1_0_textline_ori` — text line orientation
- `PP-OCRv5_server_det` — text detection
- `en_PP-OCRv5_mobile_rec` — English text recognition

---

## Ollama model management

When `--use-local` is set, the script manages the full Ollama lifecycle.

### Auto-start flow

```
Try to reach Ollama at localhost:11434
  │
  ├─ Success → check if model is available
  │              ├─ Found → ready, return
  │              └─ Not found → pull model
  │
  └─ Connection refused →
       ├─ Is `ollama` binary on PATH? (shutil.which)
       │    ├─ No → print install URL, exit
       │    └─ Yes → start `ollama serve` as background process
       │              └─ Poll localhost:11434/api/tags up to 30 times (1s apart)
       │                   ├─ Comes up → check model, pull if needed
       │                   └─ Doesn't come up → print error, exit
       │
       └─ Pull model with retry
            └─ Up to 5 attempts, exponential backoff (2s, 4s, 8s, 16s)
                 └─ Skip retries on permanent errors ("does not exist", "not found")
```

### API endpoints used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `http://localhost:11434/api/tags` | GET | List available models |
| `http://localhost:11434/api/pull` | POST | Download a model |
| `http://localhost:11434/api/generate` | POST | Run inference |

### Tags response shape

```json
{
  "models": [
    {"name": "qwen3:8b", ...},
    {"name": "llama3.1:8b", ...}
  ]
}
```

Model matching checks both exact name and base name (before the colon): `qwen3:8b` matches `qwen3:8b` exactly, or any model whose name starts with `qwen3:`.

### Pull request shape

```json
{"name": "qwen3:8b", "stream": false}
```

With `stream: false`, the response waits until the full download completes (can take minutes). Timeout is set to 600 seconds.

### Generate request shape

```json
{"model": "qwen3:8b", "prompt": "...", "stream": false}
```

Timeout: 120 seconds. Returns:

```json
{"response": "The model's text output..."}
```

### For Flutter

- If using Ollama on a local machine, the Flutter app would need to manage Ollama similarly (check, start, pull, generate)
- If using a cloud API (OpenAI), this is simpler: just make HTTP requests with the API key
- Consider offering both options in the app settings
- The Ollama auto-start logic uses `Popen` with stdout/stderr piped to DEVNULL — the process runs in the background and outlives the script

---

## OpenAI integration

### Configuration

Requires `OPENAI_API_KEY` environment variable (loaded via `python-dotenv` from `.env` file).

### API usage

Uses the official `openai` Python SDK:

```python
client = openai.OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",   # configurable via --openai-model
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
)
return resp.choices[0].message.content
```

- Temperature `0.2` is deliberately low for consistent, deterministic extraction
- The prompt is sent as a single user message (no system message)
- Only the first choice is used

### For Flutter

- Use the OpenAI Dart package or make direct HTTP requests to `https://api.openai.com/v1/chat/completions`
- Store the API key securely (not in source code)
- The same prompt text and temperature setting should produce equivalent results

---

## Publishing (GitHub Gist)

### How it works

When `--publish` is passed, the Markdown file is uploaded to a public GitHub Gist.

1. Check that the `gh` CLI is installed (`shutil.which("gh")`)
2. Run: `gh gist create --public {markdown_file_path}`
3. Parse stdout for the Gist URL
4. Print the URL to the console

### Requirements

- `gh` CLI must be installed and authenticated (`gh auth login`)
- Only the Markdown file is published (not the JSON)
- The gist is public

### For Flutter

- Use the GitHub API directly (`POST /gists`) with a personal access token
- Or offer alternative sharing methods (clipboard, share sheet, direct export)
- The Markdown format is portable and renders well on GitHub, social media, and in apps

---

## Environment dependencies

### System requirements

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| Python 3.10+ | Runtime | Yes |
| ffmpeg | Audio extraction for Whisper | Yes (checked at startup) |
| `gh` CLI | Gist publishing | Only with `--publish` |
| Ollama | Local LLM inference | Only with `--use-local` |
| `OPENAI_API_KEY` | OpenAI API access | Only without `--use-local` |

### Python packages

| Package | Purpose |
|---------|---------|
| `yt-dlp` | YouTube download |
| `faster-whisper` | Speech transcription |
| `paddleocr` + `paddlepaddle` | On-screen text OCR |
| `opencv-python` | Video frame extraction |
| `pydantic` | Data models and JSON serialization |
| `rich` | Terminal output formatting |
| `openai` | OpenAI API client |
| `python-dotenv` | Load `.env` for API keys |

### For Flutter equivalents

| Python package | Flutter/Dart equivalent to evaluate |
|----------------|-------------------------------------|
| yt-dlp | `youtube_explode_dart`, or call yt-dlp as a subprocess |
| faster-whisper | `whisper.cpp` via FFI, Google Speech-to-Text, or cloud API |
| PaddleOCR | Google ML Kit Text Recognition, Tesseract via FFI |
| OpenCV | `image` package for frame extraction, or FFmpeg via process |
| Pydantic | Dart `json_serializable` / `freezed` for data models |
| OpenAI SDK | `dart_openai` package or direct HTTP |
