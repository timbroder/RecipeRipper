# INFO_FLOW.md — Data Pipeline

How a video URL becomes a structured recipe. This document is written as a spec for reimplementing the pipeline in Dart/Flutter.

---

## High-level overview

```
Video (YouTube URL or local file)
  │
  ├─ 1. Download & metadata  ──────────► video file + title + description
  │
  ├─ 2. Listen (transcription)  ───────► raw transcript text
  │
  ├─ 3. Watch (OCR on frames)  ────────► list of on-screen text lines
  │
  ├─ 4. Extract (LLM or heuristic) ───► raw ingredients + directions
  │
  ├─ 5. Parse & clean up  ────────────► structured Ingredient objects + cleaned directions
  │
  ├─ 6. Cross-reference check  ────────► warnings about unused/missing ingredients
  │
  └─ 7. Output  ──────────────────────► JSON file + Markdown file
```

Each stage is described below with what goes in, what comes out, and how the transformation works.

---

## Stage 1: Download & metadata

### What happens

Given a YouTube URL, download the video file and extract metadata (title, description). For local files, the title comes from the filename and there is no description.

### Input

- YouTube URL (e.g. `https://www.youtube.com/shorts/ITxzpqpMZAc`) **or** a local file path

### Output

| Field | Type | Example |
|-------|------|---------|
| `video_path` | file path | `.cache/ITxzpqpMZAc.mp4` |
| `title` | string | `"Easy Garlic Pasta"` |
| `description` | string | Full YouTube description text (can be empty for local files) |

### How it works

1. Extract the 11-character video ID from the URL using regex: `(?:v=|youtu\.be/|embed/|shorts/)([\w-]{11})`
2. Check the local `.cache/` directory for a previously downloaded file (`{vid_id}.mp4`, `.mkv`, or `.webm`)
3. If cached, load the video and its `.info.json` metadata — skip the download
4. If not cached, call **yt-dlp** with:
   - Format preference: `mp4[height<=1080]/mp4/best`
   - Download subtitles (auto + manual)
   - Write info JSON
5. Save to `.cache/{vid_id}.{ext}` and `.cache/{vid_id}.info.json`
6. Return the video file path and the info dict (which contains `title`, `description`, etc.)

For **local files**, the video path is the file itself and the title is the filename stem (e.g. `pasta_recipe.mp4` → `"pasta_recipe"`). There is no description.

### Key for Dart reimplementation

- You need a YouTube downloader equivalent (yt-dlp or a Dart package)
- The caching strategy is simple file-based: check if `{id}.{ext}` exists before downloading
- The metadata JSON from yt-dlp has many fields, but only `title` and `description` are used

---

## Stage 2: Listen — speech transcription

### What happens

Extract the spoken audio from the video and transcribe it to text. This captures ingredient callouts, cooking instructions said aloud, and commentary.

### Input

| Field | Type | Notes |
|-------|------|-------|
| `video_path` | file path | The downloaded video |
| `model_size` | string | `"tiny"`, `"base"`, `"small"` (default), `"medium"`, `"large-v3"` |
| `language` | string or null | Optional 2-letter code like `"en"`, `"es"` — auto-detect if omitted |

### Output

| Field | Type | Example |
|-------|------|---------|
| `transcript` | string | `"alright so today we're making a simple garlic pasta you're gonna need about half a pound of spaghetti four cloves of garlic..."` |

### How it works

1. Load the **faster-whisper** model with:
   - Device: `cpu`
   - Compute type: `int8` (quantized for speed)
2. Transcribe the audio with `beam_size=5` and the optional language hint
3. The model returns segments, each with a `.text` field
4. Join all segment texts with spaces and trim whitespace

### Key for Dart reimplementation

- faster-whisper is a Python/C++ library. For Flutter, you'll need a Whisper binding (e.g. `whisper.cpp` via FFI, or a cloud speech-to-text API)
- The transcript is raw, unpunctuated spoken text — it includes cooking instructions mixed with commentary, opinions, and filler
- Model size affects accuracy vs. speed: `small` is the recommended default

---

## Stage 3: Watch — on-screen text extraction (OCR)

### What happens

Sample frames from the video at regular intervals and run OCR to capture any text shown on screen — ingredient lists, recipe cards, captions, step numbers, etc.

### Input

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `video_path` | file path | — | The video file |
| `seconds_between` | float | `0.6` | How often to sample a frame |
| `max_frames` | int | `180` | Cap on total frames to process |

### Output

| Field | Type | Example |
|-------|------|---------|
| `ocr_lines` | list of strings | `["1 cup flour", "2 eggs", "Preheat oven to 350F", "Subscribe!", ...]` |

### How it works

1. Open video with **OpenCV** and read the FPS
2. Calculate frame step: `step = max(1, int(seconds_between * fps))`
3. Loop through video frames, processing every `step`-th frame (up to `max_frames`):
   a. If the frame height or width is below 720 pixels, **upscale by 1.25x** using cubic interpolation (small videos produce poor OCR otherwise)
   b. Run **PaddleOCR** (English language) on the frame
   c. Extract recognized text lines that are at least 2 characters long
4. **Deduplicate** across all frames by lowercase key — the same text appearing in multiple frames is kept once
5. Return the unique text lines

### Key for Dart reimplementation

- PaddleOCR is Python-based. For Flutter, consider Google ML Kit's text recognition, Tesseract via FFI, or a cloud OCR API
- The upscaling step is important — low-resolution Shorts videos produce garbage OCR without it
- Frame sampling rate matters: 0.6s works well for Shorts (15–60 seconds). Longer videos may need wider intervals
- OCR output is noisy — it captures everything on screen including YouTube UI, usernames, subscribe buttons, and nutrition labels. The noise filtering happens in Stage 4

---

## Stage 4: Extract — turning raw text into recipe data

This is the core intelligence of the pipeline. There are two paths: **LLM-based extraction** (preferred) and **heuristic fallback**.

### Overview of the two paths

```
                    ┌─────────────────────────┐
                    │  Have LLM configured?    │
                    │  (--use-local or OpenAI) │
                    └───────┬─────────┬────────┘
                         yes│         │no
                            ▼         ▼
                    ┌───────────┐  ┌──────────────┐
                    │ LLM path  │  │ Heuristic    │
                    │           │  │ path         │
                    └───────────┘  └──────────────┘
```

### LLM path

#### Step 4a: Try description-only extraction first (YouTube only)

Before doing any transcription or OCR, the pipeline tries to extract the recipe from just the YouTube description. Many recipe videos include the full recipe in the description, which is faster and more reliable than processing audio/video.

**Input:** The YouTube description text

**The LLM prompt:**

> You are a recipe extraction assistant. Given the following video description, extract the actual recipe ingredients (with quantities when available) and cooking directions.
>
> Rules:
> - Return ONLY a JSON object with two keys: "ingredients" and "directions"
> - "ingredients" is a list of strings, each a single ingredient with quantity and unit (e.g. "1 cup flour", "2 cloves garlic, minced")
> - "directions" is a list of strings, each a single cooking step in imperative form (e.g. "Saut the onion until translucent", "Preheat oven to 350F")
> - Ignore commentary, opinions, nutrition info, and non-recipe content
> - Do NOT wrap the JSON in markdown code fences
>
> Video description:
> {description}
>
> JSON:

**Success criteria:** The result must have **at least 2 ingredients AND at least 1 direction**. If it doesn't meet this threshold, it's considered a miss and the full pipeline runs instead.

**If this succeeds:** Transcription and OCR are skipped entirely. The recipe is built from the description alone.

#### Step 4b: Full LLM extraction (all three sources)

If description-only didn't work (or this is a local file with no description), the pipeline combines all three text sources and sends them to the LLM.

**Input:**

| Source | Content |
|--------|---------|
| Transcript | Raw spoken text from Stage 2 |
| Description | YouTube description (empty for local files) |
| OCR lines | On-screen text from Stage 3, **filtered through noise detection** first |

**The LLM prompt:**

> You are a recipe extraction assistant. Given the following raw text from a recipe video, extract the actual recipe ingredients (with quantities when available) and cooking directions.
>
> Rules:
> - Return ONLY a JSON object with two keys: "ingredients" and "directions"
> - "ingredients" is a list of strings, each a single ingredient with quantity and unit (e.g. "1 cup flour", "2 cloves garlic, minced")
> - "directions" is a list of strings, each a single cooking step in imperative form (e.g. "Saut the onion until translucent", "Preheat oven to 350F")
> - Ignore commentary, opinions, nutrition info, and non-recipe content
> - Do NOT wrap the JSON in markdown code fences
>
> Raw text:
> ## Transcript
> {transcript}
>
> ## Video Description
> {description}
>
> ## On-Screen Text (OCR)
> {ocr_lines joined by newlines}
>
> JSON:

**LLM configuration:**

| Setting | OpenAI | Ollama (local) |
|---------|--------|----------------|
| Model | `gpt-4o-mini` (default) | `qwen3:8b` (default) |
| Temperature | `0.2` | default |
| Timeout | — | 120 seconds |
| API | Chat completions | `/api/generate` |

**Response parsing:**
1. Try to parse the response as JSON directly
2. If that fails, strip markdown code fences (`` ```json ... ``` ``) and retry
3. If that still fails, search for `{...}` pattern in the response and try parsing that
4. If all JSON parsing fails, fall through to the heuristic path as a fallback

**Output:** `{"ingredients": ["1 cup flour", ...], "directions": ["Preheat oven to 350F", ...]}`

### Heuristic path (fallback)

When no LLM is configured, or when LLM JSON parsing fails, the pipeline uses regex-based pattern matching to classify each line of text.

#### Noise filtering

Before classification, every line passes through a noise filter that removes:
- YouTube UI text: usernames (`@handle`), "Subscribe", "Like", "Share"
- Nutrition labels: "calories", "total fat", "serving size", "saturated fat"
- Empty section headers: "ingredients:" with no content
- Single common words: "and", "the", "a", "or"
- ALL CAPS lines of 4 words or fewer (unless they contain food words)
- OCR artifacts: long strings without spaces, single words over 12 chars
- Recipe title patterns: "cheesy/creamy/spicy X"
- Conversational fragments: "I've got the X"

#### Ingredient detection

A line is classified as an ingredient if it matches any of:
1. Has a **quantity pattern** (number like `2`, `1/2`, `1.5`) + a **food word** from a list of 95+ common ingredients
2. Has a **bullet point** (`-`, `*`) + a food word
3. Has a **unit** (cup, tbsp, oz, etc.) + a food word

Additionally, ingredients are extracted from natural speech using 6 regex patterns that handle phrases like:
- "add a can of beans" → `"1 can beans"`
- "half a cup of sugar" → `"1/2 cup sugar"`
- "2 cups flour" → `"2 cups flour"`
- "juice of one lemon" → `"juice of 1 lemon"`

#### Direction detection

A line is classified as a direction if it:
1. Starts with an **imperative cooking verb** (51 verbs: add, bake, blend, boil, chop, cook, fry, heat, mix, pour, preheat, roast, saut, season, simmer, slice, stir, whisk, etc.)
2. Contains a **temperature** mention (`350F`, `180C`)
3. Contains a **time** mention (`30 minutes`, `2 hours`)
4. Matches a numbered step pattern (`1. Preheat`, `Step 1`)
5. Contains a cooking verb + context (`cook the X`, `add until`)

Directions are also filtered to remove **commentary** — lines that sound like opinions or narration rather than instructions (e.g. "I love this recipe", "this has so much flavor", "you can use any brand").

Spoken directions are converted from descriptive to imperative form:
- "I'm gonna add the garlic" → "Add the garlic"
- "You're gonna wanna stir it" → "Stir it"
- "I've got the onions sauteing" → "Saut the onions"

#### Deduplication

Both ingredients and directions are deduplicated:
- Ingredients: normalized by removing punctuation, then checked for containment (if one ingredient's text is a subset of another, keep the longer one)
- Directions: deduplicated by lowercase match

### Output from Stage 4

| Field | Type | Example |
|-------|------|---------|
| `ing_lines` | list of strings | `["1 cup flour", "2 eggs", "1/2 tsp salt"]` |
| `dir_lines` | list of strings | `["Preheat oven to 350F", "Mix dry ingredients", "Add eggs and stir"]` |

---

## Stage 5: Parse & clean up

### Ingredient parsing

Each raw ingredient string is parsed into a structured `Ingredient` object.

**Input:** `"2 cups flour (sifted)"`

**Process:**
1. Remove bullet prefix (`-`, `*`)
2. Match quantity + unit regex at the start: `^\s*(?P<qty>FRACT)(\s*(?P<unit>UNITS_PATTERN))?\b`
   - `FRACT` matches: `1`, `1.5`, `1/2`, `1 1/2`
   - `UNITS_PATTERN` matches 38 units: tsp, tbsp, cup, oz, lb, g, kg, ml, l, clove, pinch, dash, etc.
3. Everything after the qty+unit match is the item + notes
4. If parentheses are present, extract them as notes

**Output schema:**

```json
{
  "original": "2 cups flour (sifted)",
  "quantity": "2",
  "unit": "cup",
  "item": "flour",
  "notes": "sifted"
}
```

**Pydantic model:**

```python
class Ingredient(BaseModel):
    original: str
    quantity: Optional[str] = None
    unit: Optional[str] = None
    item: Optional[str] = None
    notes: Optional[str] = None
```

### Cleanup (optional, enabled with `--cleanup`)

#### Ingredient cleanup

1. **Normalize units:** Map aliases to canonical forms
   - `teaspoon` / `teaspoons` → `tsp`
   - `tablespoon` / `tablespoons` → `tbsp`
   - `ounce` / `ounces` → `oz`
   - `pound` / `pounds` → `lb`
   - `gram` / `grams` → `g`
   - `kilogram` / `kilograms` → `kg`
   - `milliliter` / `milliliters` → `ml`
   - `liter` / `liters` → `l`
2. **Normalize Unicode fractions:** `¼` → `1/4`, `½` → `1/2`, `¾` → `3/4`, etc.
3. **Merge duplicates:** Group by `(item, unit)` and sum numeric quantities
   - `1 cup flour` + `1 cup flour` → `2 cup flour`
   - If any quantity is non-numeric, keep all entries separately

#### Direction cleanup

1. Remove time codes: `0:30`, `[1:45]`
2. Remove bullet/number prefixes: `1.`, `2)`, `-`
3. Normalize time units: `mins` → `minutes`, `hrs` → `hours`
4. Normalize temperatures: `degF` → `°F`, `degC` → `°C`
5. Apply sentence case (capitalize first letter, clean whitespace)
6. Deduplicate by lowercase
7. Merge short steps (<4 words) with the previous step

---

## Stage 6: Cross-reference check

### What happens

After extraction, the pipeline checks whether every ingredient is actually used in the directions, and whether any food words mentioned in the directions are missing from the ingredient list.

### Process

1. Extract all **food words** from the combined directions text (using a list of 95+ common foods plus multi-word items like "olive oil", "soy sauce", "cream cheese")
2. For each ingredient, extract its food words and check overlap with direction food words:
   - If no overlap → `"Unused ingredient: {item}"` warning
3. Find food words in directions that don't appear in any ingredient:
   - These become `"Missing ingredient: {word}"` warnings
   - **Auto-added** to the recipe as simple ingredients (no quantity/unit)
4. Warnings are stored on the `RecipeOutput.warnings` field

### Word normalization

Plurals are simplified for matching:
- `berries` → `berry`, `tomatoes` → `tomato`, `leaves` → `leaf`, `carrots` → `carrot`

Multi-word matches suppress their single-word components (e.g. matching "olive oil" prevents a false "oil" warning).

---

## Stage 7: Output

### Data written

Two files are created in the output directory:

#### JSON (`{slug}.json`)

Full Pydantic model dump. Example structure:

```json
{
  "title": "Easy Garlic Pasta",
  "url": "https://www.youtube.com/shorts/ITxzpqpMZAc",
  "ingredients": [
    {
      "original": "1/2 lb spaghetti",
      "quantity": "1/2",
      "unit": "lb",
      "item": "spaghetti",
      "notes": null
    }
  ],
  "directions": [
    "Bring a large pot of salted water to a boil.",
    "Cook spaghetti according to package directions."
  ],
  "extras": {
    "ocr_samples": ["1/2 lb spaghetti", "4 cloves garlic", ...]
  },
  "raw_sources": {
    "description": "Full YouTube description text (truncated to 5000 chars)...",
    "transcript": "Raw transcript text (truncated to 10000 chars)..."
  },
  "warnings": [],
  "model": "qwen3:8b"
}
```

**Full RecipeOutput schema:**

```python
class RecipeOutput(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    ingredients: List[Ingredient] = []
    directions: List[str] = []
    extras: Dict[str, List[str]] = {}      # currently just "ocr_samples"
    raw_sources: Dict[str, str] = {}        # "description" and "transcript"
    warnings: List[str] = []
    model: Optional[str] = None             # e.g. "qwen3:8b", "gpt-4o-mini", or null
```

#### Markdown (`{slug}.md`)

Human-readable recipe. Structure:

```markdown
# Recipe Extraction
**Title:** Easy Garlic Pasta
**Source:** https://www.youtube.com/shorts/ITxzpqpMZAc

## Ingredients
- 1/2 lb spaghetti
- 4 cloves garlic, minced
- 2 tbsp olive oil

## Directions
1. Bring a large pot of salted water to a boil.
2. Cook spaghetti according to package directions.
3. Meanwhile, saut garlic in olive oil over medium heat.

## Warnings
- Missing ingredient: parmesan

---
*Processed with qwen3:8b*
```

The model attribution footer only appears when an LLM was used (not in heuristic mode).

### Filename slugification

The title is converted to a filesystem-safe slug:
- Lowercase, remove non-word chars (except hyphens), replace whitespace with hyphens, collapse multiple hyphens
- `"Easy Garlic Pasta"` → `easy-garlic-pasta`
- Falls back to `"recipe"` if the title is empty or all-punctuation

### Truncation limits

| Field | Max length | Why |
|-------|-----------|-----|
| `raw_sources.description` | 5,000 chars | Prevent bloated JSON from long descriptions |
| `raw_sources.transcript` | 10,000 chars | Prevent bloated JSON from long transcripts |
| `extras.ocr_samples` | 50 entries | Prevent bloated JSON from dense OCR |
