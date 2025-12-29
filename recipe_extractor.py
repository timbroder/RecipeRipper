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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

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

# -----------------------------
# Utilities
# -----------------------------

TIMEOUT = 60 * 60  # 1 hour
YTDLP_OPTS = [
    "--no-warnings",
    "--no-call-home",
    "--ignore-errors",
]

UNITS = [
    "teaspoon","tsp","tablespoon","tbsp","cup","cups","ounce","ounces","oz",
    "pound","pounds","lb","lbs","gram","grams","g","kilogram","kilograms","kg",
    "liter","liters","l","milliliter","milliliters","ml","pinch","dash","clove","cloves",
    "slice","slices","can","cans","package","packages","stick","sticks"
]
UNITS_PATTERN = r"(?:%s)\\.?" % "|".join([re.escape(u) for u in sorted(UNITS, key=len, reverse=True)])

FRACT = r"(?:\\d+\\s*\\d?\\/\\d+|\\d+\\/\\d+|\\d+\\.\\d+|\\d+)"
QTY_PATTERN = rf"^\\s*(?P<qty>{FRACT})(\\s*(?P<unit>{UNITS_PATTERN}))?\\b"
BULLET_PATTERN = r"^\\s*[-•*]\\s*"

IMPERATIVE_VERBS = [
    "add","bake","blend","boil","braise","break","bring","broil","brown","brush","chill","chop",
    "combine","cook","cool","crack","cream","cube","cut","deglaze","dice","divide","drain","drizzle",
    "fry","fold","garnish","grate","grill","heat","knead","marinate","mash","measure","microwave",
    "mix","knead","pan-fry","pepper","pour","preheat","press","reduce","rest","roast","salt",
    "saute","sauté","score","season","sear","serve","shred","simmer","slice","soak","spoon",
    "spread","sprinkle","stir","stir-fry","strain","temper","tenderize","test","thaw",
    "toast","turn","whisk"
]
IMPERATIVE_RE = re.compile(rf'^\\s*(?:\\d+[\\).\\s-]+)?\\s*(?:{"|".join(IMPERATIVE_VERBS)})\\b', re.I)

TEMP_RE = re.compile(r"\\b(\\d{2,3})\\s*[°º]?\\s*(F|C)\\b", re.I)
TIME_RE = re.compile(r"\\b(\\d+)\\s*(seconds?|mins?|minutes?|hrs?|hours?)\\b", re.I)

def run(cmd: List[str], cwd: Optional[str]=None, timeout: int=TIMEOUT) -> subprocess.CompletedProcess:
    console.log(f"[cyan]$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)

def ensure_ffmpeg():
    try:
        cp = run(["ffmpeg","-version"], timeout=10)
        if cp.returncode != 0:
            raise RuntimeError("ffmpeg missing")
    except Exception:
        console.print("[red]ffmpeg is required. Install via brew/apt/etc.")
        sys.exit(1)

# -----------------------------
# YouTube download / metadata
# -----------------------------
def get_youtube_video_id(url: str) -> Optional[str]:
    """Extract the YouTube video ID from a URL."""
    import re
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
                    except Exception:
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
        "format": "best",
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
                except Exception:
                    pass
            return cache_path, info_json or {}
        # fallback
        for p in tmpdir.glob("*.info.json"):
            try:
                info_json = json.loads(p.read_text())
                break
            except Exception:
                pass
        return vid, info_json or {}

# -----------------------------
# Transcription (faster-whisper)
# -----------------------------
def transcribe(video_path: Path, model_size: str="large-v3", language: Optional[str]=None) -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="float16")
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
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(seconds_between * fps))
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
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
            import cv2 as _cv2
            gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            scale = 1.25 if max(h, w) < 720 else 1.0
            if scale != 1.0:
                gray = _cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=_cv2.INTER_CUBIC)
            res = ocr.ocr(gray, cls=True)
            if res and res[0]:
                for line in res[0]:
                    txt = line[1][0].strip()
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
# Parsing heuristics
# -----------------------------

def looks_like_ingredient(line: str) -> bool:
    line_s = line.strip()
    if not line_s:
        return False
    if re.search(BULLET_PATTERN, line_s):
        return True
    if re.search(QTY_PATTERN, line_s, flags=re.I):
        return True
    if re.search(UNITS_PATTERN, line_s, flags=re.I) and re.search(r"[A-Za-z]", line_s):
        return True
    if any(w in line_s.lower() for w in ["salt","pepper","oil","flour","sugar","butter","garlic","onion","egg","milk","water"]):
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
        try:
            notes = re.search(r"\\((.*?)\\)", rest).group(1)
            item = re.sub(r"\\(.*?\\)", "", rest).strip(",; ")
        except Exception:
            pass
    return Ingredient(original=line.strip(), quantity=qty, unit=unit, item=item or None, notes=notes)

def looks_like_direction(line: str) -> bool:
    ls = line.strip()
    if not ls:
        return False
    if IMPERATIVE_RE.search(ls):
        return True
    if TEMP_RE.search(ls) or TIME_RE.search(ls):
        return True
    if re.match(r"^\\s*\\d+[\\).\\s-]+", ls):
        return True
    if re.search(r"\\bstep\\s*\\d+\\b", ls, flags=re.I):
        return True
    verbs = ["cook","bake","simmer","saute","sauté","grill","roast","mix","stir","whisk","boil","fry","blend","fold"]
    if any(re.search(rf"\\b{v}\\b", ls, flags=re.I) for v in verbs):
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

def combine_sources(description: str, transcript: str, ocr_lines: List[str]) -> Tuple[List[str], List[str]]:
    text_candidates: List[str] = []
    for blob in [description or "", transcript or ""]:
        for raw in re.split(r"[\\n\\r]+", blob):
            parts = re.split(r"(?<=[\\.\\!\\?])\\s+", raw)
            for p in parts:
                p = p.strip()
                if p:
                    text_candidates.append(p)
    all_lines = ocr_lines + text_candidates
    seen = set()
    ordered = []
    for ln in all_lines:
        key = ln.strip().lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(ln)
    ing_lines, dir_lines, other = split_lines_to_sections(ordered)
    if len(ing_lines) < 3:
        for ln in other:
            if looks_like_ingredient(ln):
                ing_lines.append(ln)
    if not dir_lines and transcript:
        sentences = re.split(r"(?<=[\\.\\!\\?])\\s+", transcript)
        for s in sentences:
            if looks_like_direction(s):
                dir_lines.append(s.strip())
    return ing_lines, dir_lines

# -----------------------------
# Cleanup / Normalization (deterministic, no external APIs)
# -----------------------------

from fractions import Fraction

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
        item = re.sub(r"^\\s*[-•*]\\s*", "", item)
        item = re.sub(r"\\s+", " ", item).strip(",;: ")
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
            q_out = str(round(total, 3)).rstrip("0").rstrip(".")
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
    s = re.sub(r"[\\U00010000-\\U0010ffff]", "", s)
    s = re.sub(r"\\s+", " ", s)
    return s[0].upper() + s[1:] if s else s

def clean_directions(steps):
    out = []
    seen = set()
    for raw in steps:
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"\\(?\\b\\d{0,2}:\\d{2}\\b\\)?", "", s)
        s = re.sub(r"\\[\\s*\\d+:\\d{2}\\s*\\]", "", s)
        s = re.sub(r"^\\s*(?:[-•*]|\\d+[\\).\\s-]+)\\s*", "", s)
        s = re.sub(r"\\bmins?\\b", "minutes", s, flags=re.I)
        s = re.sub(r"\\bhrs?\\b", "hours", s, flags=re.I)
        s = re.sub(r"\\bdeg(?:rees)?\\s*F\\b", "°F", s, flags=re.I)
        s = re.sub(r"\\bdeg(?:rees)?\\s* C\\b", "°C", s, flags=re.I)
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
# Main pipeline
# -----------------------------

def process_youtube(url: str, args) -> RecipeOutput:
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        video_path, info = download_youtube(url, tmpdir)
        if not video_path or not video_path.exists():
            console.print("[red]Download failed or no video file found.")
            sys.exit(2)
        title = info.get("title")
        description = info.get("description") or ""
        transcript = transcribe(video_path, model_size=args.model, language=args.language)
        ocr_lines = ocr_onscreen_text(video_path, seconds_between=args.fps_sample, max_frames=args.max_frames)
        ing_lines, dir_lines = combine_sources(description, transcript, ocr_lines)
        ingredients = [parse_ingredient(l) for l in ing_lines]
        if args.cleanup:
            ingredients = clean_ingredients(ingredients)
        out = RecipeOutput(
            title=title,
            url=url,
            ingredients=ingredients,
            directions=clean_directions([d.strip() for d in dir_lines if d.strip()]) if args.cleanup else [d.strip() for d in dir_lines if d.strip()],
            extras={"ocr_samples": ocr_lines[:50]},
            raw_sources={"description": description[:5000], "transcript": transcript[:10000]},
        )
        return out

def process_local(video_file: str, args) -> RecipeOutput:
    vp = Path(video_file).expanduser().resolve()
    if not vp.exists():
        console.print(f"[red]Video not found: {vp}")
        sys.exit(2)
    title = vp.stem
    transcript = transcribe(vp, model_size=args.model, language=args.language)
    ocr_lines = ocr_onscreen_text(vp, seconds_between=args.fps_sample, max_frames=args.max_frames)
    ing_lines, dir_lines = combine_sources("", transcript, ocr_lines)
    ingredients = [parse_ingredient(l) for l in ing_lines]
    if args.cleanup:
        ingredients = clean_ingredients(ingredients)
    out = RecipeOutput(
        title=title,
        url=None,
        ingredients=ingredients,
        directions=clean_directions([d.strip() for d in dir_lines if d.strip()]) if args.cleanup else [d.strip() for d in dir_lines if d.strip()],
        extras={"ocr_samples": ocr_lines[:50]},
        raw_sources={"description": "", "transcript": transcript[:10000]},
    )
    return out

def save_outputs(out: RecipeOutput, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "recipe.json").write_text(out.model_dump_json(indent=2, ensure_ascii=False))
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
    (outdir / "recipe.md").write_text("\\n".join(md))

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
    console.print(table)


def preload_models(model_size: str="large-v3", lang: str="en"):
    """
    Pre-download / cache ASR (faster-whisper) and OCR (PaddleOCR) models so
    subsequent runs work fully offline.
    """
    console.print(f"[green]Preloading faster-whisper '{model_size}' and PaddleOCR ('{lang}') models…")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        fw_task = progress.add_task("Downloading faster-whisper model…", start=True)
        try:
            from faster_whisper import WhisperModel
            _ = WhisperModel(model_size, device="cpu", compute_type="float16")
            progress.update(fw_task, description="✓ faster-whisper cached")
            progress.stop_task(fw_task)
        except Exception as e:
            progress.stop_task(fw_task)
            console.print(f"[red]faster-whisper preload failed: {e}")
        ocr_task = progress.add_task("Downloading PaddleOCR model…", start=True)
        try:
            from paddleocr import PaddleOCR
            import numpy as _np
            ocr = PaddleOCR(use_angle_cls=True, lang=lang)
            dummy = (_np.zeros((64, 64, 3)) * 255).astype("uint8")
            _ = ocr.ocr(dummy, cls=True)
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
        ("small",  "~460 MB",  "~2–3 GB",   "Good balance of speed and accuracy"),
        ("medium","~1.5 GB",   "~4–6 GB",   "Better accuracy; slower"),
        ("large-v3","~3.1 GB","~8–12 GB",   "Best accuracy; recommended default"),
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
    parser.add_argument("--model", default="large-v3", help="faster-whisper model size: tiny/base/small/medium/large-v3")
    parser.add_argument("--fps-sample", type=float, default=0.6, help="Seconds between OCR frames (default 0.6s)")
    parser.add_argument("--max-frames", type=int, default=180, help="Max frames to OCR (default 180)")
    parser.add_argument("--outdir", default="output", help="Where to save recipe.json/recipe.md")
    parser.add_argument("--cleanup", action="store_true", help="Apply deterministic cleanup (normalize units, dedupe, tidy steps)")
    parser.add_argument("--preload-models", action="store_true", help="Download/cache ASR & OCR models now (offline-ready)")
    parser.add_argument("--list-models", action="store_true", help="Show recommended Faster-Whisper sizes & requirements")
    args = parser.parse_args()

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

    save_outputs(out, Path(args.outdir))
    pretty_print(out)

if __name__ == "__main__":
    main()
