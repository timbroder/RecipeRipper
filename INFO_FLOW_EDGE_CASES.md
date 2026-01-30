# INFO_FLOW_EDGE_CASES.md — Error Handling, Fallbacks & Edge Cases

How the pipeline adapts when things go wrong, data is messy, or inputs are unusual. This document covers every fallback path, error condition, and defensive measure.

---

## Pipeline-level fallbacks

### The fallback chain

The pipeline has a layered fallback strategy for recipe extraction:

```
1. Description-only LLM extraction (YouTube only)
   │
   ├─ Success (≥2 ingredients + ≥1 direction) → DONE, skip stages 2-3 entirely
   │
   └─ Fail → run full pipeline
              │
              2. Full LLM extraction (transcript + description + OCR)
              │
              ├─ Success (valid JSON) → DONE
              │
              └─ Fail (JSON parse error) → fall through
                          │
                          3. Heuristic extraction (regex-based)
                              │
                              └─ Always produces some output (may be empty lists)
```

This means:
- The **best case** is a description that already contains the recipe — no audio/video processing needed
- The **typical case** is an LLM analyzing all three text sources
- The **worst case** is pattern matching on raw text, which still produces usable results for well-structured recipes

### When description-only extraction fails

The description-only path (`llm_extract_from_description`) returns `None` when:
- The description is empty or whitespace
- The LLM returns invalid JSON
- The LLM returns fewer than 2 ingredients or 0 directions
- Any exception occurs during the LLM call

In all these cases, the pipeline proceeds to transcription + OCR + full extraction.

---

## Stage-by-stage error handling

### Stage 1: Download

| Condition | Behavior |
|-----------|----------|
| Invalid YouTube URL (no video ID found) | yt-dlp receives the raw URL and likely fails |
| yt-dlp download fails | `download_youtube` returns `(None, {})` → main prints error, exits with code 2 |
| yt-dlp returns no info JSON | Info dict is empty; title is `None`, description is `""` |
| Video format not available | yt-dlp tries fallback formats: `mp4[height<=1080]` → `mp4` → `best` |
| Cache file exists but is corrupted | The file is returned as-is; downstream stages may fail on it |
| Local video file not found | `process_local` checks `Path.exists()`, prints error, exits with code 2 |
| ffmpeg not installed | `ensure_ffmpeg()` runs at startup before any processing, exits with code 1 |

### Stage 2: Transcription

| Condition | Behavior |
|-----------|----------|
| Audio is silent or unintelligible | Whisper returns empty segments → empty transcript string |
| Wrong language detected | May produce garbled text; use `--language` to force correct language |
| Model not downloaded yet | faster-whisper auto-downloads on first use (or pre-download with `--preload-models`) |
| Video has no audio track | Whisper receives silence → empty transcript |
| Very long video | No timeout on transcription itself; the `TIMEOUT` constant (1 hour) applies to subprocess calls, not Whisper directly |

### Stage 3: OCR

| Condition | Behavior |
|-----------|----------|
| No text on screen | OCR returns empty list → pipeline relies on transcript and description |
| Low-resolution video | Frames below 720px height/width are upscaled by 1.25x before OCR |
| Non-English text on screen | PaddleOCR is configured for English; non-English text may be garbled or missed |
| Very short video (<1 second) | May produce 0-1 frames → effectively no OCR data |
| Video can't be opened by OpenCV | `cv2.VideoCapture` returns a capture object that reads no frames → empty OCR list |
| PaddleOCR returns None for a frame | The code checks `result[0] is not None` before processing boxes |

### Stage 4: LLM extraction

| Condition | Behavior |
|-----------|----------|
| Ollama not running | Auto-start flow: find binary → start `ollama serve` → poll up to 30s |
| Ollama binary not installed | Print install URL (`https://ollama.com`), exit with code 1 |
| Ollama starts but model not available | Pull model with retry (5 attempts, exponential backoff) |
| Model name doesn't exist in registry | Pull returns HTTP 500 with `"file does not exist"` → skip retries, fail immediately |
| Ollama pull fails after all retries | Print error with response body detail, exit with code 1 |
| OpenAI API key missing | `openai.OpenAI()` raises an error → unhandled (pipeline assumes key is set if not using local) |
| OpenAI rate limit / API error | Unhandled → exception propagates and script exits |
| LLM returns non-JSON text | Strip markdown fences, search for `{...}` pattern, fall back to heuristic |
| LLM returns valid JSON but wrong schema | Missing `ingredients` or `directions` key → treated as empty list via `.get()` |
| LLM returns empty lists | Pipeline continues with empty ingredient/direction lists |
| LLM timeout (120s for Ollama) | `URLError` raised → falls through to heuristic |

### Stage 5: Parsing & cleanup

| Condition | Behavior |
|-----------|----------|
| Ingredient line has no quantity | `parse_ingredient` returns `Ingredient` with `quantity=None`, `unit=None`; `item` is the full text |
| Ingredient line has no food word | Still included if it came from LLM; only filtered out in heuristic mode |
| Quantity is a Unicode fraction (`½`) | Converted to ASCII (`1/2`) during cleanup |
| Quantity can't be parsed as a number | `_to_float` returns `None`; ingredients are kept separately instead of merged |
| Two ingredients with same item but different units | Kept as separate entries (grouped by `(item, unit)` tuple) |
| Direction is very short (<4 words) | During cleanup, merged with the previous direction step |
| Direction has time codes (`[1:45]`) | Stripped during cleanup |

### Stage 6: Cross-reference

| Condition | Behavior |
|-----------|----------|
| Ingredient not mentioned in any direction | Warning: `"Unused ingredient: {item}"` |
| Food word in directions not in ingredients | Warning: `"Missing ingredient: {word}"` + auto-added to ingredients list |
| No ingredients extracted | Cross-reference runs but produces no warnings (no ingredients to check) |
| No directions extracted | Cross-reference has no direction text to scan → no missing ingredient warnings |
| Multi-word food item (e.g. "olive oil") | Matched as a phrase; suppresses false positive on single words ("oil") |
| Plural/singular mismatch | Normalized before comparison: "tomatoes" → "tomato", "berries" → "berry" |

### Stage 7: Output

| Condition | Behavior |
|-----------|----------|
| Output directory doesn't exist | Created automatically with `mkdir(parents=True, exist_ok=True)` |
| Title is empty/None | Slug falls back to `"recipe"` → files are `recipe.json` and `recipe.md` |
| Title has special characters | Slugified: remove non-word chars, replace spaces with hyphens |
| No LLM used | `model` field is `None` → no footer line in Markdown |
| `gh` CLI not installed (with `--publish`) | Print error, exit with code 1 |
| Gist creation fails | Print stderr from `gh`, exit with code 1 |

---

## Noise filtering in detail

The OCR and heuristic paths rely heavily on noise filtering to avoid including non-recipe content. Here is every noise pattern and why it exists.

### Patterns that filter YouTube UI artifacts

| Pattern | Matches | Why |
|---------|---------|-----|
| `r"^@\w+"` | `@username` | YouTube comment/creator handles |
| `r"^\d+(\.\d+)?[KMB]?\s*(views\|likes\|comments\|subscribers)"` | `1.2M views` | Video metrics |
| `r"^(subscribe\|like\|share\|follow\|comment\|tag)"` | `Subscribe!` | Call-to-action buttons |
| `r"^#\w+"` | `#cooking` | Hashtags |

### Patterns that filter nutrition information

| Pattern | Matches | Why |
|---------|---------|-----|
| `r"^(total\|saturated)\s*(fat\|carb\|sugar)"` | `total fat 5g` | Nutrition labels |
| `r"^calories"` | `Calories: 250` | Calorie counts |
| `r"^serving size"` | `Serving size: 1 cup` | Serving info |
| `r"^%\s*daily"` | `% Daily Value` | Nutrition table headers |
| `r"\b\d+\s*kcal\b"` | `250 kcal` | Calorie mentions |

### Patterns that filter empty/trivial content

| Pattern | Matches | Why |
|---------|---------|-----|
| `r"^ingredients?:?$"` | `Ingredients:` | Empty section headers |
| `r"^directions?:?$"` | `Directions` | Empty section headers |
| `r"^steps?:?$"` | `Steps:` | Empty section headers |
| `r"^(and\|the\|a\|an\|or\|but\|so\|just\|it\|is)$"` | `and` | Single common words (OCR fragments) |
| Lines < 3 chars | `hi`, `ok` | Too short to be useful |

### Patterns that filter promotional content

| Pattern | Matches | Why |
|---------|---------|-----|
| `r"^(check out\|link in\|click\|swipe\|promo\|discount\|code\|coupon\|affiliate)"` | `Link in bio` | Promotional text |

### Heuristic noise checks (not regex)

| Check | Condition | Why |
|-------|-----------|-----|
| ALL CAPS ≤4 words | `SUBSCRIBE NOW` but not `SALT AND PEPPER` | UI text is often all-caps; food words are exempted |
| Long word without spaces | `Subscribetothischannel` | OCR sometimes merges adjacent text |
| Single word >12 chars | `Antidisestablishmentarianism` | Unlikely to be recipe content |
| Recipe title pattern | `"Cheesy Garlic Bread"` | Matches "adjective + food + of/with" as a title, not an ingredient |

---

## Commentary vs. direction filtering

When processing transcripts, many spoken sentences are opinions or narration, not cooking instructions. The pipeline uses 15 commentary patterns to filter these out.

### Commentary patterns (lines are rejected if they match)

| Pattern | Example | Why |
|---------|---------|-----|
| `r"^(i mean\|yeah,\|so,\|like,\|honestly\|basically)"` | "So, this is amazing" | Conversational starters |
| `r"\b(i like it\|i love it\|i think\|in my opinion\|my favorite)"` | "I love this recipe" | Personal opinions |
| `r"\bmacros\b"` | "Great macros" | Nutrition commentary |
| `r"\bprotein and fiber\b"` | "Loaded with protein and fiber" | Nutrition commentary |
| `r"\bfirst time\b.*\bmix it with\b"` | "First time I tried to mix it with..." | Backstory |
| `r"\bthis has so much\b"` | "This has so much flavor" | Commentary |
| `r"\byou can use any\b"` | "You can use any brand" | Optional variations |
| `r"\bfollow me\b"` | "Follow me for more" | Self-promotion |
| `r"\bcheck out\b"` | "Check out my other video" | Cross-promotion |
| `r"\blet me know\b"` | "Let me know in the comments" | Engagement ask |

### Direction conversion (spoken → imperative)

Spoken transcripts use various tenses. The pipeline converts them:

| Spoken form | Imperative result |
|-------------|-------------------|
| "I'm gonna add the garlic" | "Add the garlic" |
| "You're gonna wanna stir it" | "Stir it" |
| "I've got the onions sauteing" | "Saute the onions" |
| "To the blender, I'm gonna add milk" | "Add milk to the blender" |
| "We're going to cook this for 10 minutes" | "Cook this for 10 minutes" |

Trailing fragments like "and then..." are removed. The result is capitalized and cleaned.

---

## Ingredient deduplication logic

Ingredients from multiple sources (OCR, transcript, description) often overlap. The deduplication is multi-layered:

### Step 1: Normalize

Strip punctuation, collapse multiple spaces, lowercase.

### Step 2: Containment check

If one ingredient's normalized text is a substring of another, keep only the longer one.

Example: `"flour"` and `"2 cups flour"` → keep `"2 cups flour"`

### Step 3: Food word tracking

Track which food words have been seen. If a new ingredient's food words are all already covered by existing ingredients, skip it.

Example: After seeing `"2 cups all-purpose flour"`, the ingredient `"flour"` is skipped because `"flour"` is already represented.

### Step 4: Quantity merging (during cleanup)

Group by `(item, unit)` pair and sum numeric quantities:
- `1 cup flour` + `1 cup flour` = `2 cup flour`
- `1 cup flour` + `some flour` = kept as two separate entries (can't add "some" to a number)

---

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Infrastructure error (ffmpeg missing, Ollama can't start, model pull failed, gist failed) |
| 2 | Input error (video not found, download failed) |

---

## What can produce empty output

The pipeline can complete successfully but produce an empty recipe (no ingredients, no directions) when:

1. The video has no spoken words, no on-screen text, and an empty description
2. The LLM returns empty lists (valid JSON but no content extracted)
3. In heuristic mode: no lines match ingredient or direction patterns
4. The video is not a recipe video at all

In these cases, the JSON and Markdown are still written — they just have empty ingredient and direction lists. The app should handle this gracefully (e.g. "No recipe could be extracted from this video").

---

## Concurrency and state

- The script is single-threaded and processes one video at a time
- There is no shared state between runs (each invocation is independent)
- The only side effects are:
  - Files written to `.cache/` (YouTube downloads)
  - Files written to `--outdir` (JSON + Markdown output)
  - An `ollama serve` process that may be left running in the background after auto-start
- The Ollama background process persists after the script exits — this is intentional so subsequent runs find it already running
