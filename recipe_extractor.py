#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
from rich.console import Console
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
VERBOSE = False

def vlog(msg: str) -> None:
    """Log a message only when verbose mode is enabled."""
    if VERBOSE:
        console.log(msg)

# -----------------------------
# Models / Schemas
# -----------------------------

class Ingredient(BaseModel):
    original: str
    quantity: Optional[str] = None
    unit: Optional[str] = None
    item: Optional[str] = None
    notes: Optional[str] = None

class RecipeOutput(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    ingredients: List[Ingredient] = Field(default_factory=list)
    directions: List[str] = Field(default_factory=list)
    extras: Dict[str, List[str]] = Field(default_factory=dict)
    raw_sources: Dict[str, str] = Field(default_factory=dict)  # transcript, description
    warnings: List[str] = Field(default_factory=list)
    model: Optional[str] = None

# -----------------------------
# Utilities
# -----------------------------

TIMEOUT = 60 * 60  # 1 hour

# OCR processing constants
OCR_SCALE_THRESHOLD = 720
OCR_UPSCALE_FACTOR = 1.25
QTY_PRECISION = 3
OCR_SAMPLE_LIMIT = 50
DESC_TRUNCATE_LIMIT = 5000
TRANSCRIPT_TRUNCATE_LIMIT = 10000

UNITS = [
    "teaspoon","tsp","tablespoon","tbsp","cup","cups","ounce","ounces","oz",
    "pound","pounds","lb","lbs","gram","grams","g","kilogram","kilograms","kg",
    "liter","liters","l","milliliter","milliliters","ml","pinch","dash","clove","cloves",
    "slice","slices","can","cans","package","packages","stick","sticks"
]
UNITS_PATTERN = r"(?:%s)\.?" % "|".join([re.escape(u) for u in sorted(UNITS, key=len, reverse=True)])

FRACT = r"(?:\d+\s*\d?/\d+|\d+/\d+|\d+\.\d+|\d+)"
QTY_PATTERN = rf"^\s*(?P<qty>{FRACT})(\s*(?P<unit>{UNITS_PATTERN}))?\b"
BULLET_PATTERN = r"^\s*[-•*]\s*"

IMPERATIVE_VERBS = [
    "add","bake","blend","boil","braise","break","bring","broil","brown","brush","chill","chop",
    "combine","cook","cool","crack","cream","cube","cut","deglaze","dice","divide","drain","drizzle",
    "fry","fold","garnish","grate","grill","heat","knead","marinate","mash","measure","microwave",
    "mix","pan-fry","pepper","pour","preheat","press","reduce","rest","roast","salt",
    "saute","sauté","score","season","sear","serve","shred","simmer","slice","soak","spoon",
    "spread","sprinkle","stir","stir-fry","strain","temper","tenderize","test","thaw",
    "toast","turn","whisk"
]
IMPERATIVE_RE = re.compile(rf'^\s*(?:\d+[)\.\s-]+)?\s*(?:{"|".join(IMPERATIVE_VERBS)})\b', re.I)

TEMP_RE = re.compile(r"\b(\d{2,3})\s*[°º]?\s*(F|C)\b", re.I)
TIME_RE = re.compile(r"\b(\d+)\s*(seconds?|mins?|minutes?|hrs?|hours?)\b", re.I)

def run(cmd: List[str], cwd: Optional[str]=None, timeout: int=TIMEOUT) -> subprocess.CompletedProcess:
    console.log(f"[cyan]$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)

def ensure_ffmpeg():
    try:
        cp = run(["ffmpeg","-version"], timeout=10)
        if cp.returncode != 0:
            raise RuntimeError("ffmpeg missing")
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired):
        console.print("[red]ffmpeg is required. Install via brew/apt/etc.")
        sys.exit(1)

# -----------------------------
# YouTube download / metadata
# -----------------------------
def get_youtube_video_id(url: str) -> Optional[str]:
    """Extract the YouTube video ID from a URL."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([\w-]{11})",
        r"youtube\.com/watch\?.*?v=([\w-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None

def download_youtube(url: str, tmpdir: Path) -> Tuple[Optional[Path], dict]:
    """Download YouTube video, using cache if available. Cache is always local to the project directory."""
    script_dir = Path(__file__).parent.resolve()
    cache_dir = script_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vid_id = get_youtube_video_id(url)
    cached_video = None
    cached_info = None
    if vid_id:
        for ext in ["mp4", "mkv", "webm"]:
            candidate = cache_dir / f"{vid_id}.{ext}"
            if candidate.exists():
                cached_video = candidate
                info_path = cache_dir / f"{vid_id}.info.json"
                if info_path.exists():
                    try:
                        cached_info = json.loads(info_path.read_text())
                    except (json.JSONDecodeError, OSError):
                        cached_info = None
                break
    if cached_video:
        return cached_video, cached_info or {}
    # Not cached, download
    from yt_dlp import YoutubeDL
    ytdlp_out = tmpdir / "%(id)s.%(ext)s"
    info_json: dict = {}
    ydl_opts = {
        "outtmpl": str(ytdlp_out),
        "format": "mp4[height<=1080]/mp4/best",
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en.*", "en-US"],
        "writeinfojson": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            return None, {}
        vid = None
        if "requested_downloads" in info and info["requested_downloads"]:
            entry = info["requested_downloads"][0]
            vidpath = entry.get("filepath") or entry.get("filename")
            if vidpath:
                vid = Path(vidpath)
        else:
            mp4s = list(tmpdir.glob("*.mp4"))
            if mp4s:
                vid = mp4s[0]
        # Save to cache
        if vid and vid_id:
            cache_path = cache_dir / f"{vid_id}{vid.suffix}"
            shutil.copy2(vid, cache_path)
            for p in tmpdir.glob("*.info.json"):
                try:
                    info_json = json.loads(p.read_text())
                    (cache_dir / f"{vid_id}.info.json").write_text(p.read_text())
                    break
                except (json.JSONDecodeError, OSError):
                    pass
            return cache_path, info_json or {}
        # fallback
        for p in tmpdir.glob("*.info.json"):
            try:
                info_json = json.loads(p.read_text())
                break
            except (json.JSONDecodeError, OSError):
                pass
        return vid, info_json or {}

# -----------------------------
# Transcription (faster-whisper)
# -----------------------------
def transcribe(video_path: Path, model_size: str="small", language: Optional[str]=None) -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(video_path), beam_size=5, language=language)
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())
    transcript = " ".join([t for t in text_parts if t])
    return transcript.strip()

# -----------------------------
# OCR (PaddleOCR) on sampled frames
# -----------------------------
def ocr_onscreen_text(video_path: Path, seconds_between: float=0.6, max_frames: int=180) -> List[str]:
    import cv2
    from paddleocr import PaddleOCR
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(seconds_between * fps))
    ocr = PaddleOCR(use_textline_orientation=True, lang="en")
    idx = 0
    hits: List[str] = []
    frames_sampled = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = OCR_UPSCALE_FACTOR if max(h, w) < OCR_SCALE_THRESHOLD else 1.0
            if scale != 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            res = ocr.predict(frame)
            if res and len(res) > 0 and 'rec_texts' in res[0]:
                for txt in res[0]['rec_texts']:
                    txt = txt.strip()
                    if txt and len(txt) >= 2:
                        hits.append(txt)
            frames_sampled += 1
            if frames_sampled >= max_frames:
                break
        idx += 1
    cap.release()
    seen = set()
    uniq = []
    for t in hits:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq

# -----------------------------
# LLM backend helpers
# -----------------------------

def ensure_local_model(model: str) -> None:
    """Check if an Ollama model is available locally; pull it if not."""
    url = "http://localhost:11434/api/tags"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        names = [m.get("name", "") for m in data.get("models", [])]
        # Check both exact match and match without tag
        if model in names or any(n.split(":")[0] == model.split(":")[0] for n in names):
            return
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            console.print("[red]Ollama is not installed. Get it from https://ollama.com")
            sys.exit(1)

        console.print("[yellow]Ollama is not running. Starting it automatically…")
        subprocess.Popen(
            [ollama_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        data = None
        for _ in range(30):
            time.sleep(1)
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                break
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                continue

        if data is None:
            console.print("[red]Failed to start Ollama after 30 attempts.")
            sys.exit(1)

        names = [m.get("name", "") for m in data.get("models", [])]
        if model in names or any(n.split(":")[0] == model.split(":")[0] for n in names):
            return

    console.print(f"[yellow]Pulling Ollama model '{model}' …")
    pull_url = "http://localhost:11434/api/pull"
    max_pull_attempts = 5
    last_exc: Optional[Exception] = None
    last_detail = ""
    for attempt in range(max_pull_attempts):
        if attempt > 0:
            delay = min(2 ** attempt, 16)  # 2, 4, 8, 16
            console.print(f"[yellow]Retrying pull in {delay}s (attempt {attempt + 1}/{max_pull_attempts})…")
            time.sleep(delay)
        body = json.dumps({"name": model, "stream": False}).encode()
        req = urllib.request.Request(pull_url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                resp.read()
            return
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode()
            except Exception:
                pass
            last_exc = exc
            last_detail = error_body
            console.print(f"[dim]Pull returned HTTP {exc.code}: {error_body or exc.reason}")
            # Don't retry on errors that won't resolve by waiting
            if "does not exist" in error_body or "not found" in error_body:
                break
            continue
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            last_detail = str(exc)
            continue
    console.print(f"[red]Failed to pull model '{model}': {last_exc}")
    if last_detail:
        console.print(f"[red]Detail: {last_detail}")
    sys.exit(1)


def ask_local_model(prompt: str, model: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    url = "http://localhost:11434/api/generate"
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return data.get("response", "")


def ask_openai(prompt: str, model: str) -> str:
    """Send a prompt to OpenAI and return the response text."""
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def llm_extract_recipe(
    description: str,
    transcript: str,
    ocr_lines: List[str],
    use_local: bool,
    model: str,
) -> Tuple[List[str], List[str]]:
    """Use an LLM to extract ingredients and directions from raw text sources."""
    filtered_ocr = [ln for ln in ocr_lines if not is_noise(ln)]

    parts = []
    if transcript:
        parts.append(f"## Transcript\n{transcript}")
    if description:
        parts.append(f"## Video Description\n{description}")
    if filtered_ocr:
        parts.append(f"## On-Screen Text (OCR)\n" + "\n".join(filtered_ocr))

    raw_text = "\n\n".join(parts) if parts else "(no text available)"

    prompt = (
        "You are a recipe extraction assistant. Given the following raw text from a recipe video, "
        "extract the actual recipe ingredients (with quantities when available) and cooking directions.\n\n"
        "Rules:\n"
        "- Return ONLY a JSON object with two keys: \"ingredients\" and \"directions\"\n"
        "- \"ingredients\" is a list of strings, each a single ingredient with quantity and unit "
        "(e.g. \"1 cup flour\", \"2 cloves garlic, minced\")\n"
        "- \"directions\" is a list of strings, each a single cooking step in imperative form "
        "(e.g. \"Sauté the onion until translucent\", \"Preheat oven to 350°F\")\n"
        "- Ignore commentary, opinions, nutrition info, and non-recipe content\n"
        "- Do NOT wrap the JSON in markdown code fences\n\n"
        f"Raw text:\n{raw_text}\n\n"
        "JSON:"
    )

    if use_local:
        ensure_local_model(model)
        raw = ask_local_model(prompt, model)
    else:
        raw = ask_openai(prompt, model)

    # Parse JSON from response — handle markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group())
        else:
            console.print("[yellow]LLM did not return valid JSON; falling back to heuristic parsing.")
            return combine_sources(description, transcript, ocr_lines)

    ing_lines = data.get("ingredients", [])
    dir_lines = data.get("directions", [])
    return ing_lines, dir_lines


def llm_extract_from_description(
    description: str,
    use_local: bool,
    model: str,
) -> Optional[Tuple[List[str], List[str]]]:
    """Try to extract a complete recipe from just the video description.

    Returns (ingredients, directions) if the description yields at least
    2 ingredients AND 1 direction.  Returns None otherwise (including when
    the description is empty/whitespace), so the caller can fall back to
    the full transcription + OCR pipeline.
    """
    if not description or not description.strip():
        return None

    prompt = (
        "You are a recipe extraction assistant. Given the following video description, "
        "extract the actual recipe ingredients (with quantities when available) and cooking directions.\n\n"
        "Rules:\n"
        "- Return ONLY a JSON object with two keys: \"ingredients\" and \"directions\"\n"
        "- \"ingredients\" is a list of strings, each a single ingredient with quantity and unit "
        "(e.g. \"1 cup flour\", \"2 cloves garlic, minced\")\n"
        "- \"directions\" is a list of strings, each a single cooking step in imperative form "
        "(e.g. \"Sauté the onion until translucent\", \"Preheat oven to 350°F\")\n"
        "- Ignore commentary, opinions, nutrition info, and non-recipe content\n"
        "- Do NOT wrap the JSON in markdown code fences\n\n"
        f"Video description:\n{description}\n\n"
        "JSON:"
    )

    if use_local:
        ensure_local_model(model)
        raw = ask_local_model(prompt, model)
    else:
        raw = ask_openai(prompt, model)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    ing_lines = data.get("ingredients", [])
    dir_lines = data.get("directions", [])

    if len(ing_lines) >= 2 and len(dir_lines) >= 1:
        return ing_lines, dir_lines
    return None


# -----------------------------
# Parsing heuristics
# -----------------------------

# Common food words for ingredient detection
FOOD_WORDS = [
    "salt", "pepper", "oil", "flour", "sugar", "butter", "garlic", "onion", "egg", "eggs",
    "milk", "water", "cream", "cheese", "chicken", "beef", "pork", "fish", "salmon", "shrimp",
    "tomato", "tomatoes", "potato", "potatoes", "carrot", "carrots", "celery", "broccoli",
    "spinach", "lettuce", "cucumber", "beans", "lentils", "rice", "pasta", "noodles",
    "bread", "lemon", "lime", "orange", "apple", "banana", "vinegar", "soy sauce",
    "olive oil", "vegetable oil", "coconut oil", "sesame oil", "broth", "stock",
    "wine", "beer", "honey", "maple syrup", "vanilla", "cinnamon", "cumin", "paprika",
    "oregano", "basil", "thyme", "rosemary", "parsley", "cilantro", "ginger", "turmeric",
    "nuts", "almonds", "walnuts", "cashews", "peanuts", "seeds", "yeast", "nutritional yeast",
    "tofu", "tempeh", "edamame", "mushroom", "mushrooms", "zucchini", "squash", "cabbage",
    "kale", "chard", "asparagus", "corn", "peas", "green beans", "cauliflower"
]

# Patterns that indicate YouTube UI noise or non-recipe content
NOISE_PATTERNS = [
    r"^@\w+",  # @username mentions
    r"^reply\b",  # Reply buttons
    r"^reply with video",
    r"^reply to\b",
    r"^\d+%$",  # Just percentages
    r"^%\s*\w+",  # % Daily Value etc
    r"^\d+[mg]?$",  # Just numbers or measurements without context
    r"^(total|saturated)\s*(fat|carb|sugar)",  # Nutrition label
    r"^cholesterol",
    r"^calcium",
    r"^calories",
    r"^serving size",
    r"^servings per",
    r"^daily value",
    r"^\d+\s*large\s*servings?$",
    r"^per serv",
    r"^amount",
    r"^incl\.\s*added",
    r"^made in a.*facility",
    r"^organic\s+\w+\s+beans?$",  # Product labels without quantity
    r"^ingredients?:?$",
    r"\.jpgs?$",  # Image filenames
    r"^i'm\s+\d+\s+years?\s+old",
    r"^every\s+meal",
    r"^matters\.?$",
    r"^(and|the|a|an|or|but|so|if|when|then|that|this|it|i|you|we|they|he|she)\s*$",  # Single common words
]

def is_noise(line: str) -> bool:
    """Check if a line is YouTube UI noise or non-recipe content."""
    line_s = line.strip().lower()
    if not line_s or len(line_s) < 3:
        return True
    # Check noise patterns
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, line_s, re.I):
            return True
    # Filter lines that are just usernames or hashtags
    if line_s.startswith(('#', '@')):
        return True
    # Filter lines that look like nutrition facts (multiple % or g values)
    if len(re.findall(r'\d+\s*[%g]', line_s)) > 2:
        return True
    # Filter lines that are just ALL CAPS short phrases without food words
    if line.strip().isupper() and len(line.strip().split()) <= 4:
        if not any(food in line_s for food in FOOD_WORDS):
            return True
    # Filter OCR artifacts (no spaces in long words, likely misread)
    if len(line_s) > 10 and ' ' not in line_s and not re.match(r'^[\w-]+$', line_s):
        return True
    # Filter lines that look like recipe titles (cheesy cream of X pasta)
    if re.search(r'^(cheesy|creamy|spicy|easy|quick|simple|healthy)\s+\w+\s+(of|with)\s+\w+', line_s):
        return True
    # Filter "I've got the X" type OCR fragments
    if re.search(r"^i'?v?e?\s*got\s*(the|a)?\s*\w+$", line_s):
        return True
    # Filter run-together OCR text
    words = line.strip().split()
    if len(words) == 1 and len(line_s) > 12:
        # Single "word" that's too long - likely OCR error
        return True
    return False

def looks_like_ingredient(line: str) -> bool:
    line_s = line.strip()
    if not line_s:
        return False
    # Filter out noise first
    if is_noise(line_s):
        return False
    line_lower = line_s.lower()
    # Must have a quantity pattern with a food word nearby
    if re.search(QTY_PATTERN, line_s, flags=re.I):
        if any(food in line_lower for food in FOOD_WORDS):
            return True
    # Bullet point with food word
    if re.search(BULLET_PATTERN, line_s):
        if any(food in line_lower for food in FOOD_WORDS):
            return True
    # Has unit and food word
    if re.search(UNITS_PATTERN, line_s, flags=re.I):
        if any(food in line_lower for food in FOOD_WORDS):
            return True
    return False

def parse_ingredient(line: str) -> Ingredient:
    s = re.sub(BULLET_PATTERN, "", line.strip())
    m = re.search(QTY_PATTERN, s, flags=re.I)
    qty, unit = None, None
    rest = s
    if m:
        qty = m.group("qty")
        unit = (m.group("unit") or "").strip() or None
        rest = s[m.end():].strip(",;-: \\t")
    item = rest
    notes = None
    if "(" in rest and ")" in rest:
        match = re.search(r"\((.*?)\)", rest)
        if match:
            notes = match.group(1)
            item = re.sub(r"\(.*?\)", "", rest).strip(",; ")
    return Ingredient(original=line.strip(), quantity=qty, unit=unit, item=item or None, notes=notes)

def looks_like_direction(line: str) -> bool:
    ls = line.strip()
    if not ls:
        return False
    # Filter out noise
    if is_noise(ls):
        return False
    # Allow shorter lines if they start with an imperative verb
    starts_with_imperative = IMPERATIVE_RE.search(ls)
    # Filter out very short lines unless they start with imperative
    if len(ls.split()) < 3 and not starts_with_imperative:
        return False
    # Must be a reasonable length for a direction (unless imperative)
    if len(ls) < 10 and not starts_with_imperative:
        return False
    ls_lower = ls.lower()
    # Check for imperative cooking verbs at start
    if IMPERATIVE_RE.search(ls):
        return True
    # Temperature or time references in context
    if TEMP_RE.search(ls) or TIME_RE.search(ls):
        return True
    # Numbered steps
    if re.match(r"^\s*\d+[)\.\s-]+\s*\w", ls):
        return True
    if re.search(r"\bstep\s*\d+\b", ls, flags=re.I):
        return True
    # Cooking action verbs with context (not just the word alone)
    cooking_verbs = [
        "cook", "bake", "simmer", "saute", "sauté", "grill", "roast", "mix", "stir",
        "whisk", "boil", "fry", "blend", "fold", "add", "pour", "combine", "place",
        "put", "set", "let", "allow", "wait", "remove", "take", "transfer", "serve",
        "garnish", "top", "sprinkle", "drizzle", "toss", "coat", "cover", "wrap"
    ]
    # Look for patterns like "add the X", "blend until", "cook for", etc.
    for verb in cooking_verbs:
        if re.search(rf"\b{verb}\s+(the|a|an|some|it|them|until|for|to|in|on|into|with)\b", ls_lower):
            return True
        # Also match "I'm gonna add", "just add", "then add", etc.
        if re.search(rf"\b(gonna|going to|just|then|now|first|next)\s+{verb}\b", ls_lower):
            return True
        # Match "I've got the X sauteing" or "get them all in"
        if re.search(rf"\b{verb}(s|ed|ing)?\s+(the|a|them|it|all|everything)\b", ls_lower):
            return True
    # Match "I've got the X cooking/sauteing" patterns
    if re.search(r"\b(i've got|got)\s+the\s+\w+.*\b(sauteing|cooking|boiling|simmering|frying|baking)\b", ls_lower):
        return True
    return False

def split_lines_to_sections(lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
    ing_lines, dir_lines, other = [], [], []
    for ln in lines:
        if looks_like_ingredient(ln):
            ing_lines.append(ln)
        elif looks_like_direction(ln):
            dir_lines.append(ln)
        else:
            other.append(ln)
    return ing_lines, dir_lines, other

def extract_ingredients_from_sentence(sentence: str) -> List[str]:
    """Extract all ingredient mentions from a transcript sentence like 'add a can of beans'."""
    s = sentence.lower()
    extracted = []
    seen_in_sentence = set()

    # Patterns for extracting ingredients from speech
    patterns = [
        # "add a/an/the X of Y" or "add X Y"
        r"(?:add|put|use|need|got|have|using)\s+(?:a|an|the|some)?\s*(\d*\s*(?:can|cup|tablespoon|teaspoon|tbsp|tsp|pound|lb|ounce|oz|head|bunch|clove|half|quarter)s?\s+(?:of\s+)?[\w\s]+?)(?:\.|,|$|and\b|then\b)",
        # "half a cup of X", "quarter cup of X", "quarter of a cup of X"
        r"((?:half|quarter)\s+(?:a\s+)?(?:of\s+a\s+)?(?:cup|teaspoon|tablespoon)\s+(?:of\s+)?[\w\s]+?)(?:\.|,|$|and\b)",
        # "X cups of Y", "X tablespoons Y"
        r"(\d+(?:/\d+)?\s*(?:cup|tablespoon|teaspoon|tbsp|tsp|pound|lb|ounce|oz|can|head|clove)s?\s+(?:of\s+)?[\w\s]+?)(?:\.|,|$|and\b)",
        # "the juice of X lemon"
        r"((?:the\s+)?juice\s+of\s+(?:one|two|three|a|\d+)\s+\w+)",
        # "one cup of broth", "one lemon" - capture the whole phrase
        r"((?:one|two|three|four|five)\s+(?:cup|tablespoon|teaspoon|can|head|clove|pound)s?\s+(?:of\s+)?[\w\s]+?)(?:\.|,|$|and\b)",
        # Standalone "one X" for single items
        r"(?:^|,\s*|\.\s*)((?:one|a)\s+(?:head|bunch|clove)\s+(?:of\s+)?[\w]+)",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, s)
        for match in matches:
            ingredient = match.group(1).strip()
            # Clean up the ingredient string
            ingredient = re.sub(r"\s+", " ", ingredient)
            ingredient = ingredient.strip("., ")
            # Filter out non-food matches and dedup within sentence
            key = ingredient.lower()
            if len(ingredient) > 5 and key not in seen_in_sentence:
                if any(food in ingredient for food in FOOD_WORDS):
                    seen_in_sentence.add(key)
                    extracted.append(ingredient.title())

    # Also look for "salt and pepper" type mentions
    if re.search(r"\bsalt\s+and\s+pepper\b", s) and "salt and pepper" not in seen_in_sentence:
        extracted.append("Salt And Pepper")

    return extracted

def extract_all_ingredients_from_transcript(transcript: str) -> List[str]:
    """Extract all ingredient mentions from transcript."""
    ingredients = []
    seen = set()

    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    for sentence in sentences:
        extracted = extract_ingredients_from_sentence(sentence)
        for ing in extracted:
            key = ing.lower()
            if key not in seen:
                seen.add(key)
                ingredients.append(ing)

    return ingredients

def is_actual_direction(line: str) -> bool:
    """Additional check to filter out non-directions that passed initial filter."""
    ls = line.strip().lower()
    # Filter out commentary/opinions
    commentary_patterns = [
        r"^(i mean|yeah,|so,|well,|okay,|oh,|wow|this is amazing|it's so)\b",
        r"\b(i like it|i love it|i think|i hope|helps me age|so much protein)\b",
        r"\bmacros\b",
        r"\bprotein and fiber\b",
        r"\bfirst time\b.*\bmix it with\b",
        r"\bdefinitely not\b",
        r"\bthis has so much\b",
        r"\bif you want to make this\b",
        r"\busing .* because i want\b",
        r"\bthis really helps\b",
        r"\bnutritional yeast is amazingly\b",
        r"\bso he sautes\b",
        r"\bsounds ideal\b",
        r"\byou can use any\b",
        r"\bhow long is this\b",
        r"\bput the macros\b",
        r"\bbeautiful and rich\b",
        r"\blarge servings\b",
    ]
    for pattern in commentary_patterns:
        if re.search(pattern, ls):
            return False
    return True

def convert_to_direction(sentence: str) -> str:
    """Convert a descriptive sentence to an imperative direction."""
    s = sentence.strip()
    # Convert "To the blender, I'm just gonna add X" -> "Add X to the blender"
    blender_match = re.match(r"^to the (blender|pot|pan|bowl),?\s*", s, flags=re.I)
    destination = ""
    if blender_match:
        destination = f" to the {blender_match.group(1).lower()}"
        s = s[blender_match.end():]
    # Convert "I'm gonna add X" -> "Add X"
    s = re.sub(r"^(i'm |i am )?(just )?(gonna |going to )", "", s, flags=re.I)
    # Convert "you're gonna wanna X" -> "X"
    s = re.sub(r"^(you're |you are )?(gonna |going to )?(wanna |want to )?", "", s, flags=re.I)
    # Convert "I've got the X sauteing" -> "Saute the X"
    saute_match = re.match(r"^i'?v?e?\s*got\s+the\s+(\w+),?\s*(sauteing|cooking|frying)", s, flags=re.I)
    if saute_match:
        s = f"Sauté the {saute_match.group(1)}"
    # Remove trailing "and then..."
    s = re.sub(r",?\s*and then\.\.\.?$", "", s, flags=re.I)
    # Capitalize first letter
    if s:
        s = s[0].upper() + s[1:]
    # Add destination back
    if destination and not s.endswith(destination):
        s = s.rstrip('.') + destination + "."
    return s

def combine_sources(
    description: str,
    transcript: str,
    ocr_lines: List[str],
    use_local: Optional[bool] = None,
    llm_model: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    # If LLM mode requested, delegate to llm_extract_recipe
    if use_local is not None and llm_model:
        return llm_extract_recipe(description, transcript, ocr_lines, use_local, llm_model)

    # Filter OCR lines to remove noise
    filtered_ocr = [ln for ln in ocr_lines if not is_noise(ln)]

    # Split transcript into sentences
    transcript_sentences: List[str] = []
    if transcript:
        for raw in re.split(r"[\n\r]+", transcript):
            parts = re.split(r"(?<=[\.!?])\s+", raw)
            for p in parts:
                p = p.strip()
                if p and len(p) > 10:
                    transcript_sentences.append(p)

    # Split description into sentences
    desc_sentences: List[str] = []
    if description:
        for raw in re.split(r"[\n\r]+", description):
            parts = re.split(r"(?<=[\.!?])\s+", raw)
            for p in parts:
                p = p.strip()
                if p:
                    desc_sentences.append(p)

    # Extract ingredients from transcript
    extracted_ingredients = extract_all_ingredients_from_transcript(transcript or "")

    # Combine OCR-based ingredients with transcript-extracted ingredients
    # Prioritize transcript extractions as they're cleaner
    all_ing_candidates = extracted_ingredients.copy()

    # Add OCR lines that look like ingredients (with food words)
    for ln in filtered_ocr:
        if looks_like_ingredient(ln):
            all_ing_candidates.append(ln)

    # Add description lines that look like ingredients
    for ln in desc_sentences:
        if looks_like_ingredient(ln):
            all_ing_candidates.append(ln)

    # Deduplicate ingredients with fuzzy matching for partial matches
    seen = set()
    seen_foods = set()  # Track which food items we've already captured
    ing_lines = []
    for ln in all_ing_candidates:
        key = ln.strip().lower()
        # Normalize key for dedup
        key_normalized = re.sub(r'[^\w\s]', '', key)  # Remove punctuation
        key_normalized = re.sub(r'\s+', ' ', key_normalized).strip()
        if not key_normalized:
            continue
        # Check if this is a subset/duplicate of existing ingredient
        is_dup = False
        for existing in seen:
            # If new key is contained in existing or vice versa, it's a dup
            if key_normalized in existing or existing in key_normalized:
                is_dup = True
                break
        # Also check if main food word already captured
        for food in FOOD_WORDS:
            if food in key_normalized and food in seen_foods:
                # Check if this is just a partial match (like "CUP OF BROTH" when we have "one cup of broth")
                if len(key_normalized.split()) <= 3:
                    is_dup = True
                    break
        if not is_dup and key_normalized not in seen:
            seen.add(key_normalized)
            # Track food words in this ingredient
            for food in FOOD_WORDS:
                if food in key_normalized:
                    seen_foods.add(food)
            ing_lines.append(ln)

    # Extract directions from transcript - only actual cooking steps
    dir_lines = []
    seen_dirs = set()
    for sentence in transcript_sentences:
        if looks_like_direction(sentence) and is_actual_direction(sentence):
            direction = convert_to_direction(sentence)
            key = direction.lower()
            if key not in seen_dirs:
                seen_dirs.add(key)
                dir_lines.append(direction)

    # Also check description for directions
    for sentence in desc_sentences:
        if looks_like_direction(sentence) and is_actual_direction(sentence):
            direction = convert_to_direction(sentence)
            key = direction.lower()
            if key not in seen_dirs:
                seen_dirs.add(key)
                dir_lines.append(direction)

    return ing_lines, dir_lines

# -----------------------------
# Cleanup / Normalization (deterministic, no external APIs)
# -----------------------------

UNIT_ALIASES = {
    "teaspoon": "tsp", "teaspoons": "tsp", "tsp.": "tsp", "tsp": "tsp",
    "tablespoon": "tbsp", "tablespoons": "tbsp", "tbsp.": "tbsp", "tbsp": "tbsp",
    "cup": "cup", "cups": "cup",
    "ounce": "oz", "ounces": "oz", "oz.": "oz", "oz": "oz",
    "pound": "lb", "pounds": "lb", "lb.": "lb", "lbs": "lb", "lb": "lb",
    "gram": "g", "grams": "g", "g.": "g", "g": "g",
    "kilogram": "kg", "kilograms": "kg", "kg.": "kg", "kg": "kg",
    "milliliter": "ml", "milliliters": "ml", "ml.": "ml", "ml": "ml",
    "liter": "l", "liters": "l", "l.": "l", "l": "l",
    "clove": "clove", "cloves": "clove",
    "slice": "slice", "slices": "slice",
    "can": "can", "cans": "can",
    "stick": "stick", "sticks": "stick",
    "package": "package", "packages": "package",
    "pinch": "pinch", "dash": "dash"
}

UNICODE_FRACTIONS = {
    "¼": "1/4", "½": "1/2", "¾": "3/4",
    "⅐": "1/7", "⅑": "1/9", "⅒": "1/10",
    "⅓": "1/3", "⅔": "2/3",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
    "⅙": "1/6", "⅚": "5/6",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}

def _normalize_unicode_fracs(text: str) -> str:
    for u, ascii_frac in UNICODE_FRACTIONS.items():
        text = text.replace(u, ascii_frac)
    return text

def _to_float(qty: str):
    if not qty:
        return None
    s = _normalize_unicode_fracs(qty).strip()
    try:
        if " " in s and "/" in s:
            a, b = s.split(" ", 1)
            return float(Fraction(a)) + float(Fraction(b))
        if "/" in s:
            return float(Fraction(s))
        return float(s)
    except Exception:
        return None

def _canon_unit(unit):
    if not unit:
        return None
    key = unit.strip().lower().strip(".")
    return UNIT_ALIASES.get(key, key)

def clean_ingredients(ings):
    norm = []
    for ing in ings:
        qty = ing.quantity
        unit = _canon_unit(ing.unit)
        item = (ing.item or ing.original).strip()
        item = re.sub(r"^\s*[-•*]\s*", "", item)
        item = re.sub(r"\s+", " ", item).strip(",;: ")
        norm.append(Ingredient(
            original=ing.original,
            quantity=_normalize_unicode_fracs(qty) if qty else None,
            unit=unit,
            item=item,
            notes=ing.notes
        ))
    buckets = {}
    ordered_keys = []
    for ing in norm:
        key = (ing.item.lower() if ing.item else "", ing.unit or "")
        if key not in buckets:
            buckets[key] = []
            ordered_keys.append(key)
        buckets[key].append(ing)
    merged = []
    for key in ordered_keys:
        items = buckets[key]
        total = 0.0
        all_numeric = True
        for it in items:
            f = _to_float(it.quantity) if it.quantity else None
            if f is None:
                all_numeric = False
                break
            total += f
        if all_numeric and items[0].unit:
            q_out = str(round(total, QTY_PRECISION)).rstrip("0").rstrip(".")
            merged.append(Ingredient(
                original=items[0].original,
                quantity=q_out,
                unit=items[0].unit,
                item=items[0].item,
                notes=None
            ))
        else:
            merged.append(items[0])
            for it in items[1:]:
                merged.append(it)
    return merged

def _sentence_case(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s[0].upper() + s[1:] if s else s

def clean_directions(steps):
    out = []
    seen = set()
    for raw in steps:
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"\(?\b\d{0,2}:\d{2}\b\)?", "", s)
        s = re.sub(r"\[\s*\d+:\d{2}\s*\]", "", s)
        s = re.sub(r"^\s*(?:[-•*]|\d+[)\.\s-]+)\s*", "", s)
        s = re.sub(r"\bmins?\b", "minutes", s, flags=re.I)
        s = re.sub(r"\bhrs?\b", "hours", s, flags=re.I)
        s = re.sub(r"\bdeg(?:rees)?\s*F\b", "°F", s, flags=re.I)
        s = re.sub(r"\bdeg(?:rees)?\s*C\b", "°C", s, flags=re.I)
        s = _sentence_case(s)
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    merged = []
    for step in out:
        if merged and len(step.split()) < 4:
            merged[-1] = (merged[-1].rstrip(".") + " " + step).strip()
        else:
            merged.append(step)
    return merged

# -----------------------------
# Cross-reference check
# -----------------------------

def _normalize_word(word: str) -> str:
    """Simple plural stripping for food-word matching."""
    w = word.lower().strip()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("ves"):
        return w[:-3] + "f"
    if len(w) > 2 and w.endswith("es"):
        stem = w[:-2]
        if stem.endswith(("ch", "sh", "ss")):
            return w[:-1]
        if stem.endswith(("c", "g", "x", "z")):
            return w[:-1]
        return stem
    if len(w) > 1 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _extract_food_words_from_text(text: str) -> set:
    """Return normalized food words found in *text*."""
    text_lower = text.lower()
    found = set()
    # Multi-word food items: exact phrase match
    for fw in FOOD_WORDS:
        if " " in fw and re.search(r"\b" + re.escape(fw) + r"\b", text_lower):
            found.add(_normalize_word(fw))
    # Single-word food items: normalize every token so plurals match
    text_normalized = {_normalize_word(w) for w in re.findall(r"\b\w+\b", text_lower)}
    for fw in FOOD_WORDS:
        if " " not in fw and _normalize_word(fw) in text_normalized:
            found.add(_normalize_word(fw))
    # Remove single-word items that are part of a matched multi-word item
    multi_word_foods = {_normalize_word(fw) for fw in FOOD_WORDS if " " in fw and _normalize_word(fw) in found}
    for mw in multi_word_foods:
        for part in mw.split():
            normed = _normalize_word(part)
            if normed in found and normed != mw:
                found.discard(normed)
    return found


def cross_reference_check(recipe: RecipeOutput) -> List[str]:
    """Flag unused ingredients and missing ingredients."""
    warnings: List[str] = []
    if not recipe.ingredients or not recipe.directions:
        return warnings

    directions_text = " ".join(recipe.directions)
    directions_food = _extract_food_words_from_text(directions_text)

    all_ingredient_food: set = set()
    for ing in recipe.ingredients:
        item_text = ing.item or ing.original
        item_food = _extract_food_words_from_text(item_text)
        all_ingredient_food.update(item_food)
        if item_food and not (item_food & directions_food):
            warnings.append(f"Unused ingredient: {item_text}")

    missing = directions_food - all_ingredient_food
    for word in sorted(missing):
        warnings.append(f"Missing ingredient: {word} (added to ingredients)")
        recipe.ingredients.append(Ingredient(original=word, item=word))

    recipe.warnings = warnings
    return warnings


# -----------------------------
# Main pipeline
# -----------------------------

def _process_directions(dir_lines: List[str], cleanup: bool) -> List[str]:
    """Process direction lines with optional cleanup."""
    stripped = [d.strip() for d in dir_lines if d.strip()]
    return clean_directions(stripped) if cleanup else stripped


def process_youtube(url: str, args) -> RecipeOutput:
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        vlog(f"[cyan]Downloading YouTube video: {url}")
        video_path, info = download_youtube(url, tmpdir)
        if not video_path or not video_path.exists():
            console.print("[red]Download failed or no video file found.")
            sys.exit(2)
        title = info.get("title")
        description = info.get("description") or ""
        vlog(f"[cyan]Title: {title}")
        vlog(f"[cyan]Description length: {len(description)} chars")
        use_local = getattr(args, "use_local", False)
        llm_model = getattr(args, "local_model", None) if use_local else getattr(args, "openai_model", None)
        vlog(f"[cyan]LLM mode: {'local (' + llm_model + ')' if use_local and llm_model else ('openai (' + llm_model + ')' if llm_model else 'off (heuristic)')}")

        # Try extracting from description alone when LLM mode is active
        desc_result = None
        if llm_model:
            vlog("[cyan]Attempting description-first extraction…")
            desc_result = llm_extract_from_description(description, use_local, llm_model)
            if desc_result is None:
                vlog("[yellow]Description-first extraction returned no complete recipe.")
            else:
                vlog(f"[green]Description-first extraction found {len(desc_result[0])} ingredients, {len(desc_result[1])} directions.")

        if desc_result is not None:
            ing_lines, dir_lines = desc_result
            transcript = ""
            ocr_lines: List[str] = []
            console.log("[green]Recipe extracted from description — skipping video transcription/OCR.")
        else:
            vlog("[cyan]Starting transcription…")
            transcript = transcribe(video_path, model_size=args.model, language=args.language)
            vlog(f"[cyan]Transcription done — {len(transcript)} chars.")
            vlog("[cyan]Starting OCR…")
            ocr_lines = ocr_onscreen_text(video_path, seconds_between=args.fps_sample, max_frames=args.max_frames)
            vlog(f"[cyan]OCR done — {len(ocr_lines)} unique lines.")
            vlog("[cyan]Combining sources…")
            ing_lines, dir_lines = combine_sources(description, transcript, ocr_lines, use_local=use_local, llm_model=llm_model)

        vlog(f"[cyan]Extracted {len(ing_lines)} ingredient lines, {len(dir_lines)} direction lines.")
        ingredients = [parse_ingredient(l) for l in ing_lines]
        if args.cleanup:
            vlog("[cyan]Running cleanup…")
            ingredients = clean_ingredients(ingredients)
        out = RecipeOutput(
            title=title,
            url=url,
            ingredients=ingredients,
            directions=_process_directions(dir_lines, args.cleanup),
            extras={"ocr_samples": ocr_lines[:OCR_SAMPLE_LIMIT]},
            raw_sources={"description": description[:DESC_TRUNCATE_LIMIT], "transcript": transcript[:TRANSCRIPT_TRUNCATE_LIMIT]},
            model=llm_model,
        )
        return out

def process_local(video_file: str, args) -> RecipeOutput:
    vp = Path(video_file).expanduser().resolve()
    if not vp.exists():
        console.print(f"[red]Video not found: {vp}")
        sys.exit(2)
    title = vp.stem
    vlog(f"[cyan]Processing local video: {vp}")
    vlog("[cyan]Starting transcription…")
    transcript = transcribe(vp, model_size=args.model, language=args.language)
    vlog(f"[cyan]Transcription done — {len(transcript)} chars.")
    vlog("[cyan]Starting OCR…")
    ocr_lines = ocr_onscreen_text(vp, seconds_between=args.fps_sample, max_frames=args.max_frames)
    vlog(f"[cyan]OCR done — {len(ocr_lines)} unique lines.")
    use_local = getattr(args, "use_local", False)
    llm_model = getattr(args, "local_model", None) if use_local else getattr(args, "openai_model", None)
    vlog(f"[cyan]LLM mode: {'local (' + llm_model + ')' if use_local and llm_model else ('openai (' + llm_model + ')' if llm_model else 'off (heuristic)')}")
    vlog("[cyan]Combining sources…")
    ing_lines, dir_lines = combine_sources("", transcript, ocr_lines, use_local=use_local, llm_model=llm_model)
    vlog(f"[cyan]Extracted {len(ing_lines)} ingredient lines, {len(dir_lines)} direction lines.")
    ingredients = [parse_ingredient(l) for l in ing_lines]
    if args.cleanup:
        vlog("[cyan]Running cleanup…")
        ingredients = clean_ingredients(ingredients)
    out = RecipeOutput(
        title=title,
        url=None,
        ingredients=ingredients,
        directions=_process_directions(dir_lines, args.cleanup),
        extras={"ocr_samples": ocr_lines[:OCR_SAMPLE_LIMIT]},
        raw_sources={"description": "", "transcript": transcript[:TRANSCRIPT_TRUNCATE_LIMIT]},
        model=llm_model,
    )
    return out

def _slugify(text: str) -> str:
    """Turn a title into a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-")
    return s or "recipe"


def save_outputs(out: RecipeOutput, outdir: Path) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(out.title) if out.title else "recipe"
    json_path = outdir / f"{slug}.json"
    md_path = outdir / f"{slug}.md"
    json_path.write_text(out.model_dump_json(indent=2, ensure_ascii=False))
    md = ["# Recipe Extraction"]
    if out.title:
        md.append(f"**Title:** {out.title}")
    if out.url:
        md.append(f"**Source:** {out.url}")
    md.append("")
    md.append("## Ingredients")
    if out.ingredients:
        for ing in out.ingredients:
            md.append(f"- {ing.quantity + ' ' if ing.quantity else ''}{(ing.unit + ' ') if ing.unit else ''}{ing.item or ing.original}")
    else:
        md.append("_None detected_")
    md.append("")
    md.append("## Directions")
    if out.directions:
        for i, step in enumerate(out.directions, 1):
            md.append(f"{i}. {step}")
    else:
        md.append("_None detected_")
    if out.warnings:
        md.append("")
        md.append("## Warnings")
        for w in out.warnings:
            md.append(f"- {w}")
    if out.model:
        md.append("")
        md.append("---")
        md.append(f"*Processed with {out.model}*")
    md_path.write_text("\n".join(md))
    return json_path, md_path

def publish_gist(files: List[Path]) -> str:
    """Upload files to a public GitHub Gist via the gh CLI. Returns the Gist URL."""
    if not shutil.which("gh"):
        console.print("[red]The 'gh' CLI is required for --publish. Install from https://cli.github.com/")
        sys.exit(1)
    result = subprocess.run(
        ["gh", "gist", "create", "--public", *[str(f) for f in files]],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Failed to create gist: {result.stderr.strip()}")
        sys.exit(1)
    gist_url = result.stdout.strip()
    console.print(f"[green]Gist created: {gist_url}")
    return gist_url


def pretty_print(out: RecipeOutput):
    table = Table(title="Extracted Recipe", box=box.SIMPLE, show_lines=False)
    table.add_column("Section", style="bold cyan", no_wrap=True)
    table.add_column("Content", style="")
    if out.ingredients:
        ing_txt = "\n".join([f"- {i.quantity + ' ' if i.quantity else ''}{(i.unit + ' ') if i.unit else ''}{i.item or i.original}" for i in out.ingredients[:20]])
        if len(out.ingredients) > 20:
            ing_txt += f"\n… (+{len(out.ingredients)-20} more)"
    else:
        ing_txt = "None"
    table.add_row("Ingredients", ing_txt)
    if out.directions:
        dir_txt = "\n".join([f"{i+1}. {s}" for i, s in enumerate(out.directions[:12])])
        if len(out.directions) > 12:
            dir_txt += f"\n… (+{len(out.directions)-12} more)"
    else:
        dir_txt = "None"
    table.add_row("Directions", dir_txt)
    if out.warnings:
        warn_txt = "\n".join(f"- {w}" for w in out.warnings)
        table.add_row("Warnings", warn_txt, style="yellow")
    console.print(table)


def preload_models(model_size: str="small", lang: str="en"):
    """
    Pre-download / cache ASR (faster-whisper) and OCR (PaddleOCR) models so
    subsequent runs work fully offline.
    """
    console.print(f"[green]Preloading faster-whisper '{model_size}' and PaddleOCR ('{lang}') models…")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        fw_task = progress.add_task("Downloading faster-whisper model…", start=True)
        try:
            from faster_whisper import WhisperModel
            _ = WhisperModel(model_size, device="cpu", compute_type="int8")
            progress.update(fw_task, description="✓ faster-whisper cached")
            progress.stop_task(fw_task)
        except Exception as e:
            progress.stop_task(fw_task)
            console.print(f"[red]faster-whisper preload failed: {e}")
        ocr_task = progress.add_task("Downloading PaddleOCR model…", start=True)
        try:
            from paddleocr import PaddleOCR
            import numpy as _np
            ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
            dummy = (_np.zeros((64, 64, 3)) * 255).astype("uint8")
            _ = ocr.predict(dummy)
            progress.update(ocr_task, description="✓ PaddleOCR cached")
            progress.stop_task(ocr_task)
        except Exception as e:
            progress.stop_task(ocr_task)
            console.print(f"[red]PaddleOCR preload failed: {e}")


def list_models():
    """
    Print recommended Faster-Whisper model sizes with rough download size and typical RAM/VRAM needs.
    """
    from rich.table import Table
    from rich.console import Console
    c = Console()
    t = Table(title="Faster-Whisper Model Options", show_lines=False)
    t.add_column("Flag", style="bold cyan")
    t.add_column("Approx. Download")
    t.add_column("Typical CPU RAM")
    t.add_column("Notes")
    rows = [
        ("tiny",   "~75 MB",   "~0.5–1 GB", "Fastest, lower accuracy; good for quick drafts"),
        ("base",   "~145 MB",  "~1–2 GB",   "Balanced speed/accuracy for Shorts"),
        ("small",  "~460 MB",  "~2–3 GB",   "Recommended default"),
        ("medium","~1.5 GB",   "~4–6 GB",   "Better accuracy; slower"),
        ("large-v3","~3.1 GB","~8–12 GB",   "Best accuracy; slowest on CPU"),
    ]
    for r in rows:
        t.add_row(*r)
    c.print(t)


def main():
    parser = argparse.ArgumentParser(description="Extract ingredients & directions from a recipe video (YouTube or local).")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--youtube", help="YouTube URL (works with Shorts)")
    g.add_argument("--video", help="Path to a local video file")
    parser.add_argument("--language", help="Language hint (e.g., en, es, fr). Auto-detect if omitted.")
    parser.add_argument("--model", default="small", help="faster-whisper model size: tiny/base/small/medium/large-v3")
    parser.add_argument("--fps-sample", type=float, default=0.6, help="Seconds between OCR frames (default 0.6s)")
    parser.add_argument("--max-frames", type=int, default=180, help="Max frames to OCR (default 180)")
    parser.add_argument("--outdir", default="output", help="Where to save recipe.json/recipe.md")
    parser.add_argument("--cleanup", action="store_true", help="Apply deterministic cleanup (normalize units, dedupe, tidy steps)")
    parser.add_argument("--use-local", action="store_true", help="Use a local Ollama model instead of OpenAI for recipe extraction")
    parser.add_argument("--local-model", default="qwen3:8b", help="Ollama model name (default: qwen3:8b)")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model name (default: gpt-4o-mini)")
    parser.add_argument("--preload-models", action="store_true", help="Download/cache ASR & OCR models now (offline-ready)")
    parser.add_argument("--list-models", action="store_true", help="Show recommended Faster-Whisper sizes & requirements")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of each pipeline step")
    parser.add_argument("--publish", action="store_true", help="Upload output files to a public GitHub Gist (requires gh CLI)")
    parser.add_argument("--paprika", action="store_true", help="Copy Gist URL to clipboard and open Paprika (implies --publish)")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    ensure_ffmpeg()

    if args.list_models:
        list_models()
        if not (args.youtube or args.video or args.preload_models):
            return

    if args.preload_models:
        preload_models(model_size=args.model, lang=(args.language or 'en'))
        console.print('[bold green]Preload complete. You can run offline next time.')
        if not (args.youtube or args.video):
            return

    if args.youtube:
        out = process_youtube(args.youtube, args)
    else:
        out = process_local(args.video, args)

    warnings = cross_reference_check(out)
    if warnings:
        vlog(f"[yellow]Cross-reference check found {len(warnings)} warning(s).")

    saved = save_outputs(out, Path(args.outdir))
    pretty_print(out)
    if args.paprika:
        args.publish = True
    if args.publish:
        gist_url = publish_gist([saved[1]])
        if args.paprika:
            subprocess.run(["bash", "-c", f'echo -n "{gist_url}" | pbcopy && open -a "Paprika Recipe Manager 3"'])
            console.print("[green]Gist URL copied to clipboard. Paste it into Paprika's browser.")

if __name__ == "__main__":
    main()
