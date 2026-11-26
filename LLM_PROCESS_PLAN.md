# Plan: Replace Heuristic Parsing with LLM-based Recipe Extraction

## Summary

Replace the regex/heuristic-based `combine_sources()` parsing with an LLM call that extracts structured recipe data from raw transcript + OCR text. Support both OpenAI and local Ollama models, following the same pattern as ModelTagger.

## Files to Modify

- `recipe_extractor.py` — main changes
- `requirements.txt` — add `openai`
- `tests/test_recipe_extractor.py` — update tests

## Changes

### 1. CLI Arguments (in `build_parser()`)

Add to the argument parser:
- `--use-local` — `store_true`, use Ollama instead of OpenAI
- `--local-model` — default `"llama3.1:8b-instruct"`, Ollama model name
- `--openai-model` — default `"gpt-4o-mini"`, OpenAI model name

### 2. LLM Backend Functions (new section after OCR section)

Following ModelTagger's pattern:

- `ensure_local_model(model)` — check/pull Ollama model via `http://localhost:11434/api/tags` and `/api/pull`
- `ask_local_model(prompt, model)` — POST to `http://localhost:11434/api/generate`, return response text
- `ask_openai(prompt, model)` — use `openai.OpenAI()` client with chat completions, return response text

### 3. LLM Recipe Extraction Function

New function `llm_extract_recipe(description, transcript, ocr_lines, use_local, model)`:

- Build a prompt with the raw text sources (description, transcript, filtered OCR)
- Prompt instructs the LLM to return JSON with `{"ingredients": ["..."], "directions": ["..."]}`
- Call `ask_local_model()` or `ask_openai()` based on `use_local`
- Parse the JSON response
- Return `(ing_lines, dir_lines)` tuple (same shape as `combine_sources`)

The prompt will:
- Provide the transcript, description, and OCR text
- Ask the LLM to extract only actual recipe ingredients (with quantities) and cooking directions
- Ask for imperative form directions (e.g., "Sauté the onion" not "I've got the onion sauteing")
- Request JSON output format

### 4. Modify `combine_sources()` Signature

Add optional LLM params: `combine_sources(description, transcript, ocr_lines, use_local=None, llm_model=None)`

- If `use_local is not None` (i.e., LLM is available), call `llm_extract_recipe()`
- Otherwise, fall back to the existing heuristic parsing (preserve as fallback)

### 5. Update `process_youtube()` and `process_local()`

Pass `args.use_local` and the appropriate model name through to `combine_sources()`.

### 6. Update `requirements.txt`

Add `openai` package.

### 7. Update Tests

- Add test for `ask_openai()` with mocked OpenAI client
- Add test for `ask_local_model()` with mocked requests
- Add test for `llm_extract_recipe()` with mocked LLM response
- Update `test_process_youtube` and `test_process_local_success` to pass new args
- Keep existing heuristic parsing tests (they still test the fallback path)

## Verification

1. `python -m pytest tests/ -v` — all tests pass
2. `python recipe_extractor.py --youtube "https://www.youtube.com/shorts/K6wEWWhJf7Q" --outdir output_test --cleanup` — OpenAI extraction (requires OPENAI_API_KEY)
3. `python recipe_extractor.py --youtube "https://www.youtube.com/shorts/K6wEWWhJf7Q" --outdir output_test --cleanup --use-local` — Ollama extraction (requires running Ollama)
4. Verify output has clean ingredients list and ordered directions
