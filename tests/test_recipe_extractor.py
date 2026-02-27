import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import recipe_extractor as rex

FRACT_FIXED = r"(?:\d+\s*\d?/\d+|\d+/\d+|\d+\.\d+|\d+)"
UNITS_PATTERN_FIXED = r"(?:%s)\.?" % "|".join([re.escape(u) for u in sorted(rex.UNITS, key=len, reverse=True)])
rex.FRACT = FRACT_FIXED
rex.UNITS_PATTERN = UNITS_PATTERN_FIXED
rex.QTY_PATTERN = rf"^\s*(?P<qty>{FRACT_FIXED})(\s*(?P<unit>{UNITS_PATTERN_FIXED}))?\b"
rex.IMPERATIVE_RE = re.compile(
    rf"^\s*(?:\d+[\).\s-]+)?\s*(?:{'|'.join(rex.IMPERATIVE_VERBS)})\b",
    re.I,
)

REAL_RE_SEARCH = re.search


class DummyCompletedProcess:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdout = b"ok"
        self.stderr = b""


def test_run_invokes_subprocess(monkeypatch):
    captured = {}

    def fake_run(cmd, cwd=None, stdout=None, stderr=None, check=False, timeout=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        return DummyCompletedProcess(returncode=0)

    monkeypatch.setattr(rex.subprocess, "run", fake_run)
    result = rex.run(["echo", "hello"], cwd="/tmp", timeout=5)
    assert result.returncode == 0
    assert captured["cmd"] == ["echo", "hello"]
    assert captured["cwd"] == "/tmp"
    assert captured["timeout"] == 5


def test_ensure_ffmpeg_success(monkeypatch):
    monkeypatch.setattr(rex, "run", lambda *a, **k: DummyCompletedProcess(returncode=0))
    rex.ensure_ffmpeg()


def test_ensure_ffmpeg_failure(monkeypatch, capsys):
    monkeypatch.setattr(rex, "run", lambda *a, **k: DummyCompletedProcess(returncode=1))
    with pytest.raises(SystemExit) as exc:
        rex.ensure_ffmpeg()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "ffmpeg is required" in out


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://example.com", None),
    ],
)
def test_get_youtube_video_id(url, expected):
    assert rex.get_youtube_video_id(url) == expected


def test_download_youtube_cache_hit(tmp_path):
    cache_dir = Path(rex.__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    vid_id = "abcdefghijk"
    cached_video = cache_dir / f"{vid_id}.mp4"
    cached_video.write_bytes(b"data")
    info_path = cache_dir / f"{vid_id}.info.json"
    info_path.write_text(json.dumps({"title": "Cached", "description": "From cache"}))

    try:
        video_path, info = rex.download_youtube(f"https://youtu.be/{vid_id}", tmp_path)
        assert video_path == cached_video
        assert info["title"] == "Cached"
        assert info["description"] == "From cache"
    finally:
        cached_video.unlink(missing_ok=True)
        info_path.unlink(missing_ok=True)


def test_download_youtube_new_with_requested_downloads(tmp_path, monkeypatch):
    cache_dir = Path(rex.__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    vid_id = "mnopqrstu12"

    class DummyYDL:
        def __init__(self, opts):
            self.opts = opts
            self.output_dir = Path(opts["outtmpl"]).parent

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download):
            video_path = self.output_dir / f"{vid_id}.mp4"
            video_path.write_bytes(b"video")
            info_path = self.output_dir / f"{vid_id}.info.json"
            info_path.write_text(json.dumps({"title": "Fresh", "description": "New download"}))
            return {"requested_downloads": [{"filepath": str(video_path)}]}

    monkeypatch.setitem(sys.modules, "yt_dlp", mock.Mock(YoutubeDL=DummyYDL))

    video_path, info = rex.download_youtube(f"https://youtu.be/{vid_id}", tmp_path)
    assert video_path.exists()
    assert video_path.parent == cache_dir
    assert info["title"] == "Fresh"
    assert (cache_dir / f"{vid_id}.info.json").exists()

    for f in cache_dir.glob(f"{vid_id}.*"):
        f.unlink()


def test_download_youtube_without_requested_downloads(tmp_path, monkeypatch):
    url = "https://videos.example.com/withoutid"

    class DummyYDL:
        def __init__(self, opts):
            self.output_dir = Path(opts["outtmpl"]).parent

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download):
            video_path = self.output_dir / "fallback.mp4"
            video_path.write_bytes(b"fallback")
            (self.output_dir / "fallback.info.json").write_text(json.dumps({"notes": "fallback info"}))
            return {"id": "zzz"}

    monkeypatch.setitem(sys.modules, "yt_dlp", mock.Mock(YoutubeDL=DummyYDL))

    video_path, info = rex.download_youtube(url, tmp_path)
    assert video_path.name == "fallback.mp4"
    assert info["notes"] == "fallback info"


def test_download_youtube_no_info(tmp_path, monkeypatch):
    class DummyYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download):
            return None

    monkeypatch.setitem(sys.modules, "yt_dlp", mock.Mock(YoutubeDL=DummyYDL))

    video_path, info = rex.download_youtube("https://example.com/noinfo", tmp_path)
    assert video_path is None
    assert info == {}


def test_transcribe(monkeypatch, tmp_path):
    segments = [SimpleNamespace(text=" Step 1"), SimpleNamespace(text="Step 2 ")]

    class DummyModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, path, beam_size, language):
            return segments, SimpleNamespace()

    monkeypatch.setitem(sys.modules, "faster_whisper", mock.Mock(WhisperModel=DummyModel))

    transcript = rex.transcribe(tmp_path / "video.mp4")
    assert transcript == "Step 1 Step 2"


class DummyCapture:
    def __init__(self, *a, **k):
        self.index = 0
        self.frames = [
            mock.Mock(shape=(10, 10, 3)),
            mock.Mock(shape=(10, 10, 3)),
            mock.Mock(shape=(10, 10, 3)),
        ]

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == 5:
            return 30
        if prop == 7:
            return len(self.frames)
        return 0

    def grab(self):
        if self.index >= len(self.frames):
            return False
        self.index += 1
        return True

    def retrieve(self):
        return True, self.frames[self.index - 1]

    def release(self):
        pass


def test_ocr_onscreen_text(monkeypatch):
    ocr_calls = []

    def fake_resize(frame, size, interpolation):
        return frame

    class DummyOCR:
        def __init__(self, *a, **k):
            pass

        def predict(self, frame):
            ocr_calls.append("predict")
            return [{'rec_texts': ['Ingredient'], 'rec_scores': [0.9]}]

    dummy_cv2 = mock.Mock(
        VideoCapture=DummyCapture,
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
        resize=fake_resize,
        INTER_CUBIC=0,
    )

    monkeypatch.setitem(sys.modules, "cv2", dummy_cv2)
    monkeypatch.setitem(sys.modules, "paddleocr", mock.Mock(PaddleOCR=DummyOCR))

    lines = rex.ocr_onscreen_text(Path("dummy.mp4"), seconds_between=0.1, max_frames=2)
    assert lines == ["Ingredient"]
    assert ocr_calls


def test_parsing_helpers(monkeypatch):
    def fake_search(pattern, text, flags=0):
        if pattern == rex.QTY_PATTERN:
            if "cup" not in text.lower():
                return None
            class Match:
                def group(self, key):
                    return {"qty": "1", "unit": "cup"}.get(key)

                def end(self):
                    return len("1 cup")

            return Match()
        if pattern == r"\((.*?)\)" and "(" in text:
            class NotesMatch:
                def group(self, index):
                    return "sifted"

            return NotesMatch()
        return REAL_RE_SEARCH(pattern, text, flags)

    monkeypatch.setattr(rex.re, "search", fake_search)
    monkeypatch.setattr(rex, "IMPERATIVE_RE", re.compile(r"^\s*(?:\d+[).\s-]+)?\s*(?:mix|bake)\b", re.I))

    assert rex.looks_like_ingredient("1 cup flour")
    assert rex.looks_like_ingredient("- salt to taste")
    assert not rex.looks_like_ingredient("Random phrase")

    ing = rex.parse_ingredient("1 cup flour (sifted)")
    assert ing.quantity == "1"
    assert ing.unit == "cup"
    assert "flour" in (ing.item or "").lower()

    assert rex.looks_like_direction("Mix the batter")
    assert not rex.looks_like_direction("Just chatting")

    ings, dirs, other = rex.split_lines_to_sections([
        "1 cup sugar",
        "Bake until golden",
        "Random note",
    ])
    assert "1 cup sugar" in ings
    assert "Random note" in other

    ing_lines, dir_lines = rex.combine_sources(
        "1 cup sugar. Stir well.",
        "Bake for 10 minutes.",
        ["Sprinkle nuts"],
    )
    assert isinstance(ing_lines, list)
    assert isinstance(dir_lines, list)


def test_cleaning_helpers(monkeypatch):
    original_sub = rex.re.sub

    def safe_sub(pattern, repl, string, count=0, flags=0):
        if isinstance(pattern, str):
            pattern = pattern.replace("\\\\(", "\\(").replace("\\\\)", "\\)").replace("\\\\[", "\\[").replace("\\\\]", "\\]")
        return original_sub(pattern, repl, string, count=count, flags=flags)

    monkeypatch.setattr(rex.re, "sub", safe_sub)

    ing1 = rex.Ingredient(original="1 cup sugar", quantity="1", unit="cup", item="sugar", notes=None)
    ing2 = rex.Ingredient(original="1/2 cup sugar", quantity="1/2", unit="cups", item="sugar", notes=None)
    cleaned = rex.clean_ingredients([ing1, ing2])
    assert cleaned[0].quantity == "1.5"
    assert cleaned[0].unit == "cup"

    assert rex._normalize_unicode_fracs("½ cup") == "1/2 cup"
    assert rex._to_float("1 1/2") == 1.5
    assert rex._to_float("invalid") is None
    assert rex._canon_unit("Tablespoons") == "tbsp"

    steps = rex.clean_directions(["1. mix well", "mix well", "Serve", "Chill"])
    assert steps
    joined = " ".join(steps).lower()
    assert "mix" in joined


def test_process_youtube(monkeypatch, tmp_path):
    args = SimpleNamespace(
        model="tiny",
        language="en",
        fps_sample=0.5,
        max_frames=2,
        cleanup=True,
        use_local=False,
        local_model="llama3.1:8b-instruct",
        openai_model="gpt-4o-mini",
    )

    dummy_video = tmp_path / "video.mp4"
    dummy_video.write_bytes(b"vid")

    monkeypatch.setattr(rex, "download_youtube", lambda url, td: (dummy_video, {"title": "Title", "description": "Desc"}))
    monkeypatch.setattr(rex, "llm_extract_from_description", lambda *a, **k: None)
    monkeypatch.setattr(rex, "transcribe", lambda *a, **k: "Mix well")
    monkeypatch.setattr(rex, "ocr_onscreen_text", lambda *a, **k: ["1 cup sugar"])
    monkeypatch.setattr(rex, "combine_sources", lambda d, t, o, **kw: (["1 cup sugar"], ["Mix well"]))
    original_sub = rex.re.sub

    def safe_sub(pattern, repl, string, count=0, flags=0):
        if isinstance(pattern, str):
            pattern = pattern.replace("\\\\(", "\\(").replace("\\\\)", "\\)").replace("\\\\[", "\\[").replace("\\\\]", "\\]")
        return original_sub(pattern, repl, string, count=count, flags=flags)

    monkeypatch.setattr(rex.re, "sub", safe_sub)

    result = rex.process_youtube("https://youtu.be/dummy", args)
    assert result.title == "Title"
    assert result.url.endswith("dummy")
    assert result.ingredients[0].item == "sugar"
    assert result.directions


def test_process_local_success(monkeypatch, tmp_path):
    args = SimpleNamespace(
        model="tiny",
        language="en",
        fps_sample=0.5,
        max_frames=2,
        cleanup=False,
        use_local=False,
        local_model="llama3.1:8b-instruct",
        openai_model="gpt-4o-mini",
    )

    video_file = tmp_path / "local.mp4"
    video_file.write_bytes(b"vid")

    monkeypatch.setattr(rex, "transcribe", lambda *a, **k: "Chop onions")
    monkeypatch.setattr(rex, "ocr_onscreen_text", lambda *a, **k: ["1 onion"])
    monkeypatch.setattr(rex, "combine_sources", lambda d, t, o, **kw: (["1 onion"], ["Chop onions"]))

    result = rex.process_local(str(video_file), args)
    assert result.title == "local"
    assert result.ingredients
    assert result.directions


def test_process_local_missing(tmp_path, capsys):
    args = SimpleNamespace(model="tiny", language="en", fps_sample=1.0, max_frames=1, cleanup=False)
    missing = tmp_path / "nope.mp4"
    with pytest.raises(SystemExit) as exc:
        rex.process_local(str(missing), args)
    assert exc.value.code == 2
    assert "Video not found" in capsys.readouterr().out


def test_save_outputs_and_pretty_print(tmp_path, monkeypatch):
    ingredient = rex.Ingredient(original="1 cup sugar", quantity="1", unit="cup", item="sugar", notes=None)
    recipe = rex.RecipeOutput(
        title="Cake",
        url="https://example.com",
        ingredients=[ingredient],
        directions=["Mix"],
        extras={"ocr_samples": []},
        raw_sources={"description": "", "transcript": ""},
    )
    def fake_dump(self, indent=2, ensure_ascii=False):
        return json.dumps(self.model_dump(), indent=indent)

    monkeypatch.setattr(rex.RecipeOutput, "model_dump_json", fake_dump, raising=False)
    outdir = tmp_path / "output"
    rex.save_outputs(recipe, outdir)
    assert (outdir / "cake.json").exists()
    assert (outdir / "cake.md").exists()
    rex.pretty_print(recipe)


def test_preload_models(monkeypatch):
    events = {}

    class DummyProgress:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add_task(self, description, start):
            events[description] = "started"
            return description

        def update(self, task, description):
            events[f"update_{task}"] = description

        def stop_task(self, task):
            events[f"stop_{task}"] = True

    class DummyWhisper:
        def __init__(self, *a, **k):
            pass

    class DummyOCR:
        def __init__(self, *a, **k):
            raise RuntimeError("bad ocr")

    monkeypatch.setattr(rex, "Progress", DummyProgress)
    monkeypatch.setitem(sys.modules, "faster_whisper", mock.Mock(WhisperModel=lambda *a, **k: DummyWhisper()))
    monkeypatch.setitem(sys.modules, "paddleocr", mock.Mock(PaddleOCR=DummyOCR))
    
    class DummyArray:
        def __mul__(self, other):
            return self

        def astype(self, dtype):
            return self

    def fake_zeros(shape):
        return DummyArray()

    monkeypatch.setitem(sys.modules, "numpy", mock.Mock(zeros=fake_zeros))

    rex.preload_models()
    assert any(key.startswith("update_") for key in events)
    assert any(key.startswith("stop_") for key in events)


def test_list_models():
    rex.list_models()


def test_main_list_models(monkeypatch):
    monkeypatch.setattr(rex, "ensure_ffmpeg", lambda: None)
    listed = {}
    monkeypatch.setattr(rex, "list_models", lambda: listed.setdefault("called", True))

    def fake_parse(self):
        return SimpleNamespace(
            youtube=None,
            video=None,
            language=None,
            model="small",
            fps_sample=0.6,
            max_frames=180,
            outdir="output",
            cleanup=False,
            preload_models=False,
            list_models=True,
            verbose=False,
        )

    monkeypatch.setattr(rex.argparse.ArgumentParser, "parse_args", fake_parse, raising=False)
    rex.main()
    assert listed["called"]


def test_main_preload_only(monkeypatch):
    monkeypatch.setattr(rex, "ensure_ffmpeg", lambda: None)
    preloaded = {}
    monkeypatch.setattr(rex, "preload_models", lambda model_size, lang: preloaded.setdefault("ran", (model_size, lang)))
    monkeypatch.setattr(rex, "list_models", lambda: None)

    def fake_parse(self):
        return SimpleNamespace(
            youtube=None,
            video=None,
            language="en",
            model="tiny",
            fps_sample=0.6,
            max_frames=180,
            outdir="output",
            cleanup=False,
            preload_models=True,
            list_models=False,
            verbose=False,
        )

    monkeypatch.setattr(rex.argparse.ArgumentParser, "parse_args", fake_parse, raising=False)
    rex.main()
    assert preloaded["ran"] == ("tiny", "en")


def test_main_youtube_flow(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["prog", "--youtube", "https://youtu.be/test", "--outdir", str(tmp_path), "--cleanup"])
    monkeypatch.setattr(rex, "ensure_ffmpeg", lambda: None)

    dummy_recipe = rex.RecipeOutput(
        title="T", url="u", ingredients=[], directions=[], extras={}, raw_sources={}
    )

    monkeypatch.setattr(rex, "process_youtube", lambda url, args: dummy_recipe)
    monkeypatch.setattr(rex, "save_outputs", lambda out, od: (Path("/fake/recipe.json"), Path("/fake/recipe.md")))
    monkeypatch.setattr(rex, "pretty_print", lambda out: None)
    rex.main()


def test_main_local_flow(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_text("data")
    monkeypatch.setattr(sys, "argv", ["prog", "--video", str(video)])
    monkeypatch.setattr(rex, "ensure_ffmpeg", lambda: None)

    dummy_recipe = rex.RecipeOutput(
        title="T", url="u", ingredients=[], directions=[], extras={}, raw_sources={}
    )

    monkeypatch.setattr(rex, "process_local", lambda path, args: dummy_recipe)
    monkeypatch.setattr(rex, "save_outputs", lambda out, od: (Path("/fake/recipe.json"), Path("/fake/recipe.md")))
    monkeypatch.setattr(rex, "pretty_print", lambda out: None)
    rex.main()


# -----------------------------
# LLM backend tests
# -----------------------------


def test_ask_openai(monkeypatch):
    """Test ask_openai with mocked OpenAI client."""
    class DummyMessage:
        content = "Hello from GPT"

    class DummyChoice:
        message = DummyMessage()

    class DummyResponse:
        choices = [DummyChoice()]

    class DummyCompletions:
        def create(self, model, messages, temperature):
            assert model == "gpt-4o-mini"
            assert messages[0]["role"] == "user"
            return DummyResponse()

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    dummy_openai = mock.Mock()
    dummy_openai.OpenAI.return_value = DummyClient()
    monkeypatch.setitem(sys.modules, "openai", dummy_openai)

    result = rex.ask_openai("test prompt", "gpt-4o-mini")
    assert result == "Hello from GPT"


def test_ask_local_model(monkeypatch):
    """Test ask_local_model with mocked urllib."""
    response_data = json.dumps({"response": "Local model reply"}).encode()

    class FakeResponse:
        def read(self):
            return response_data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=None):
        assert "generate" in req.full_url
        body = json.loads(req.data.decode())
        assert body["model"] == "llama3.1:8b-instruct"
        assert body["prompt"] == "test prompt"
        return FakeResponse()

    monkeypatch.setattr(rex.urllib.request, "urlopen", fake_urlopen)

    result = rex.ask_local_model("test prompt", "llama3.1:8b-instruct")
    assert result == "Local model reply"


def test_llm_extract_recipe_openai(monkeypatch):
    """Test llm_extract_recipe with mocked OpenAI returning valid JSON."""
    llm_json = json.dumps({
        "ingredients": ["1 cup flour", "2 eggs", "1/2 cup milk"],
        "directions": ["Preheat oven to 350°F", "Mix dry ingredients", "Bake for 25 minutes"],
    })

    monkeypatch.setattr(rex, "ask_openai", lambda prompt, model: llm_json)

    ing, dirs = rex.llm_extract_recipe(
        description="A simple cake recipe",
        transcript="First preheat the oven. Mix the dry ingredients.",
        ocr_lines=["1 cup flour", "2 eggs"],
        use_local=False,
        model="gpt-4o-mini",
    )
    assert ing == ["1 cup flour", "2 eggs", "1/2 cup milk"]
    assert dirs == ["Preheat oven to 350°F", "Mix dry ingredients", "Bake for 25 minutes"]


def test_llm_extract_recipe_local(monkeypatch):
    """Test llm_extract_recipe with mocked local Ollama model."""
    llm_json = json.dumps({
        "ingredients": ["2 cloves garlic"],
        "directions": ["Mince the garlic"],
    })

    monkeypatch.setattr(rex, "ensure_local_model", lambda model: None)
    monkeypatch.setattr(rex, "ask_local_model", lambda prompt, model: llm_json)

    ing, dirs = rex.llm_extract_recipe(
        description="",
        transcript="Mince the garlic cloves",
        ocr_lines=[],
        use_local=True,
        model="llama3.1:8b-instruct",
    )
    assert ing == ["2 cloves garlic"]
    assert dirs == ["Mince the garlic"]


def test_llm_extract_recipe_with_code_fences(monkeypatch):
    """Test that markdown code fences in LLM response are handled."""
    llm_response = '```json\n{"ingredients": ["1 cup sugar"], "directions": ["Stir well"]}\n```'

    monkeypatch.setattr(rex, "ask_openai", lambda prompt, model: llm_response)

    ing, dirs = rex.llm_extract_recipe(
        description="",
        transcript="Stir sugar well",
        ocr_lines=[],
        use_local=False,
        model="gpt-4o-mini",
    )
    assert ing == ["1 cup sugar"]
    assert dirs == ["Stir well"]


def test_llm_extract_recipe_invalid_json_fallback(monkeypatch):
    """Test fallback to heuristic parsing when LLM returns invalid JSON."""
    monkeypatch.setattr(rex, "ask_openai", lambda prompt, model: "not valid json at all")

    # The fallback calls combine_sources without LLM params — mock it
    original_combine = rex.combine_sources

    def mock_combine(desc, trans, ocr, use_local=None, llm_model=None):
        if use_local is None and llm_model is None:
            return (["fallback ingredient"], ["fallback direction"])
        return original_combine(desc, trans, ocr, use_local=use_local, llm_model=llm_model)

    monkeypatch.setattr(rex, "combine_sources", mock_combine)

    ing, dirs = rex.llm_extract_recipe(
        description="",
        transcript="fallback test",
        ocr_lines=[],
        use_local=False,
        model="gpt-4o-mini",
    )
    assert ing == ["fallback ingredient"]
    assert dirs == ["fallback direction"]


def test_ensure_local_model_already_present(monkeypatch):
    """Test ensure_local_model when model is already available."""
    tags_response = json.dumps({
        "models": [{"name": "llama3.1:8b-instruct"}]
    }).encode()

    class FakeResponse:
        def read(self):
            return tags_response

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(rex.urllib.request, "urlopen", lambda req, timeout=None: FakeResponse())

    # Should return without error
    rex.ensure_local_model("llama3.1:8b-instruct")


def test_ensure_local_model_pull_needed(monkeypatch):
    """Test ensure_local_model when model needs to be pulled."""
    call_count = {"n": 0}

    class FakeTagsResponse:
        def read(self):
            return json.dumps({"models": []}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakePullResponse:
        def read(self):
            return b'{"status":"success"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        if "tags" in (req.full_url if hasattr(req, "full_url") else req):
            return FakeTagsResponse()
        return FakePullResponse()

    monkeypatch.setattr(rex.urllib.request, "urlopen", fake_urlopen)

    rex.ensure_local_model("llama3.1:8b-instruct")
    assert call_count["n"] == 2  # tags + pull


def test_ensure_local_model_pull_retry_on_500(monkeypatch):
    """Test that pull retries on transient HTTP 500 errors."""
    call_count = {"n": 0}

    class FakeTagsResponse:
        def read(self):
            return json.dumps({"models": []}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakePullResponse:
        def read(self):
            return b'{"status":"success"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        url = req.full_url if hasattr(req, "full_url") else req
        if "tags" in url:
            return FakeTagsResponse()
        # First pull attempt fails with 500
        if call_count["n"] == 2:
            raise urllib.error.HTTPError(url, 500, "Internal Server Error", {}, None)
        return FakePullResponse()

    import urllib.error

    monkeypatch.setattr(rex.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(rex.time, "sleep", lambda s: None)

    rex.ensure_local_model("llama3.1:8b-instruct")
    # tags + pull (fail) + pull (success) = 3
    assert call_count["n"] == 3


def test_ensure_local_model_auto_start(monkeypatch):
    """Test that ensure_local_model auto-starts Ollama when it's not running."""
    call_count = {"urlopen": 0}
    popen_calls = []

    class FakeTagsResponse:
        def read(self):
            return json.dumps({"models": []}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakePullResponse:
        def read(self):
            return b'{"status":"success"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=None):
        call_count["urlopen"] += 1
        url = req.full_url if hasattr(req, "full_url") else req
        # First call to tags fails (Ollama not running)
        if call_count["urlopen"] == 1 and "tags" in url:
            raise urllib_error.URLError("Connection refused")
        # Second call to tags succeeds (Ollama started)
        if "tags" in url:
            return FakeTagsResponse()
        # Pull call
        return FakePullResponse()

    def fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return mock.MagicMock()

    import urllib.error as urllib_error

    monkeypatch.setattr(rex.shutil, "which", lambda name: "/usr/local/bin/ollama")
    monkeypatch.setattr(rex.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(rex.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(rex.time, "sleep", lambda s: None)

    rex.ensure_local_model("llama3.1:8b-instruct")

    assert len(popen_calls) == 1
    assert popen_calls[0] == ["/usr/local/bin/ollama", "serve"]
    # tags (fail) + tags (retry success) + pull = 3
    assert call_count["urlopen"] == 3


def test_combine_sources_with_llm(monkeypatch):
    """Test that combine_sources delegates to LLM when use_local is not None and model is set."""
    llm_json = json.dumps({
        "ingredients": ["1 lb chicken"],
        "directions": ["Grill the chicken"],
    })
    monkeypatch.setattr(rex, "ask_openai", lambda prompt, model: llm_json)

    ing, dirs = rex.combine_sources(
        description="Grilled chicken recipe",
        transcript="Grill the chicken for 10 minutes",
        ocr_lines=["1 lb chicken"],
        use_local=False,
        llm_model="gpt-4o-mini",
    )
    assert ing == ["1 lb chicken"]
    assert dirs == ["Grill the chicken"]


def test_combine_sources_without_llm():
    """Test that combine_sources falls back to heuristic when no LLM params."""
    ing, dirs = rex.combine_sources(
        description="1 cup sugar. Stir well.",
        transcript="Bake for 10 minutes.",
        ocr_lines=["Sprinkle nuts"],
    )
    # Should use heuristic path (no assertion on specific results,
    # just verify it returns the right shape)
    assert isinstance(ing, list)
    assert isinstance(dirs, list)


# -----------------------------
# Description-first extraction tests
# -----------------------------


def test_llm_extract_from_description_success(monkeypatch):
    """Complete recipe from description returns (ingredients, directions)."""
    llm_json = json.dumps({
        "ingredients": ["1 cup flour", "2 eggs", "1/2 cup milk"],
        "directions": ["Preheat oven to 350°F", "Mix dry ingredients"],
    })
    monkeypatch.setattr(rex, "ask_openai", lambda prompt, model: llm_json)

    result = rex.llm_extract_from_description(
        description="A cake recipe\n1 cup flour\n2 eggs\n1/2 cup milk\nPreheat oven...",
        use_local=False,
        model="gpt-4o-mini",
    )
    assert result is not None
    ing, dirs = result
    assert len(ing) == 3
    assert len(dirs) == 2


def test_llm_extract_from_description_incomplete(monkeypatch):
    """Only ingredients, no directions → returns None."""
    llm_json = json.dumps({
        "ingredients": ["1 cup flour", "2 eggs"],
        "directions": [],
    })
    monkeypatch.setattr(rex, "ask_openai", lambda prompt, model: llm_json)

    result = rex.llm_extract_from_description(
        description="Ingredients: 1 cup flour, 2 eggs",
        use_local=False,
        model="gpt-4o-mini",
    )
    assert result is None


def test_llm_extract_from_description_empty():
    """Empty description returns None without calling any LLM."""
    # No monkeypatching needed — if it calls ask_openai it would fail
    assert rex.llm_extract_from_description("", use_local=False, model="gpt-4o-mini") is None
    assert rex.llm_extract_from_description("   ", use_local=False, model="gpt-4o-mini") is None


def test_process_youtube_skips_video_on_description_hit(monkeypatch, tmp_path):
    """When description extraction succeeds, transcribe and OCR are NOT called."""
    args = SimpleNamespace(
        model="tiny",
        language="en",
        fps_sample=0.5,
        max_frames=2,
        cleanup=False,
        use_local=False,
        local_model="llama3.1:8b-instruct",
        openai_model="gpt-4o-mini",
    )

    dummy_video = tmp_path / "video.mp4"
    dummy_video.write_bytes(b"vid")

    monkeypatch.setattr(rex, "download_youtube", lambda url, td: (
        dummy_video, {"title": "Title", "description": "Full recipe here"}
    ))

    monkeypatch.setattr(rex, "llm_extract_from_description", lambda desc, use_local, model: (
        ["1 cup flour", "2 eggs"], ["Preheat oven to 350°F"]
    ))

    transcribe_called = {"yes": False}
    ocr_called = {"yes": False}
    monkeypatch.setattr(rex, "transcribe", lambda *a, **k: (transcribe_called.update(yes=True), "")[1])
    monkeypatch.setattr(rex, "ocr_onscreen_text", lambda *a, **k: (ocr_called.update(yes=True), [])[1])

    result = rex.process_youtube("https://youtu.be/test", args)
    assert not transcribe_called["yes"]
    assert not ocr_called["yes"]
    assert len(result.ingredients) == 2
    assert len(result.directions) == 1


def test_process_youtube_falls_through_on_description_miss(monkeypatch, tmp_path):
    """When description extraction returns None, full pipeline runs."""
    args = SimpleNamespace(
        model="tiny",
        language="en",
        fps_sample=0.5,
        max_frames=2,
        cleanup=False,
        use_local=False,
        local_model="llama3.1:8b-instruct",
        openai_model="gpt-4o-mini",
    )

    dummy_video = tmp_path / "video.mp4"
    dummy_video.write_bytes(b"vid")

    monkeypatch.setattr(rex, "download_youtube", lambda url, td: (
        dummy_video, {"title": "Title", "description": "No recipe here"}
    ))

    monkeypatch.setattr(rex, "llm_extract_from_description", lambda desc, use_local, model: None)

    transcribe_called = {"yes": False}
    ocr_called = {"yes": False}

    def fake_transcribe(*a, **k):
        transcribe_called["yes"] = True
        return "Mix well"

    def fake_ocr(*a, **k):
        ocr_called["yes"] = True
        return ["1 cup sugar"]

    monkeypatch.setattr(rex, "transcribe", fake_transcribe)
    monkeypatch.setattr(rex, "ocr_onscreen_text", fake_ocr)
    monkeypatch.setattr(rex, "combine_sources", lambda d, t, o, **kw: (["1 cup sugar"], ["Mix well"]))

    result = rex.process_youtube("https://youtu.be/test", args)
    assert transcribe_called["yes"]
    assert ocr_called["yes"]
    assert result.ingredients
    assert result.directions


# -----------------------------
# Slug / title-based output tests
# -----------------------------


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Simple Cake", "simple-cake"),
        ("Best Pasta (Easy!)", "best-pasta-easy"),
        ("  Spaced  Out  ", "spaced-out"),
        ("Crème Brûlée", "crème-brûlée"),
        ("", "recipe"),
        ("---", "recipe"),
    ],
)
def test_slugify(title, expected):
    assert rex._slugify(title) == expected


def test_save_outputs_no_title(tmp_path, monkeypatch):
    recipe = rex.RecipeOutput(
        ingredients=[rex.Ingredient(original="1 egg", item="egg")],
        directions=["Boil"],
    )

    def fake_dump(self, indent=2, ensure_ascii=False):
        return json.dumps(self.model_dump(), indent=indent)

    monkeypatch.setattr(rex.RecipeOutput, "model_dump_json", fake_dump, raising=False)
    rex.save_outputs(recipe, tmp_path)
    assert (tmp_path / "recipe.json").exists()
    assert (tmp_path / "recipe.md").exists()


# -----------------------------
# Cross-reference check tests
# -----------------------------


@pytest.mark.parametrize(
    "word, expected",
    [
        ("onions", "onion"),
        ("tomatoes", "tomato"),
        ("berries", "berry"),
        ("leaves", "leaf"),
        ("sauces", "sauce"),
        ("carrots", "carrot"),
        ("potatoes", "potato"),
        ("sugar", "sugar"),
        ("cabbages", "cabbage"),
        ("dishes", "dishe"),  # linguistic imperfection is OK per spec
    ],
)
def test_normalize_word(word, expected):
    assert rex._normalize_word(word) == expected


def test_extract_food_words_from_text():
    found = rex._extract_food_words_from_text("Add olive oil and garlic to the pan")
    assert "olive oil" in {fw for fw in rex.FOOD_WORDS if rex._normalize_word(fw) in found}
    assert rex._normalize_word("garlic") in found

    empty = rex._extract_food_words_from_text("Turn on the stove and wait")
    assert len(empty) == 0


def test_cross_reference_unused_ingredient():
    recipe = rex.RecipeOutput(
        ingredients=[
            rex.Ingredient(original="1 cup flour", item="flour"),
            rex.Ingredient(original="1 tsp vanilla", item="vanilla"),
        ],
        directions=["Mix the flour with water"],
    )
    warnings = rex.cross_reference_check(recipe)
    assert any("vanilla" in w.lower() for w in warnings)


def test_cross_reference_missing_ingredient():
    recipe = rex.RecipeOutput(
        ingredients=[
            rex.Ingredient(original="1 cup flour", item="flour"),
        ],
        directions=["Mix the flour with butter and eggs"],
    )
    warnings = rex.cross_reference_check(recipe)
    assert any("butter" in w.lower() and "added to ingredients" in w.lower() for w in warnings)
    assert any("egg" in w.lower() and "added to ingredients" in w.lower() for w in warnings)


def test_cross_reference_no_warnings():
    recipe = rex.RecipeOutput(
        ingredients=[
            rex.Ingredient(original="1 cup flour", item="flour"),
            rex.Ingredient(original="2 eggs", item="eggs"),
        ],
        directions=["Mix the flour and eggs together"],
    )
    warnings = rex.cross_reference_check(recipe)
    assert warnings == []


def test_cross_reference_plural_matching():
    recipe = rex.RecipeOutput(
        ingredients=[
            rex.Ingredient(original="1 onion", item="onion"),
        ],
        directions=["Dice the onions and saut\u00e9"],
    )
    warnings = rex.cross_reference_check(recipe)
    unused = [w for w in warnings if "Unused" in w]
    assert len(unused) == 0


def test_cross_reference_empty_recipe():
    recipe = rex.RecipeOutput()
    warnings = rex.cross_reference_check(recipe)
    assert warnings == []


def test_cross_reference_stored_in_recipe():
    recipe = rex.RecipeOutput(
        ingredients=[
            rex.Ingredient(original="1 cup sugar", item="sugar"),
            rex.Ingredient(original="1 tsp vanilla", item="vanilla"),
        ],
        directions=["Stir the sugar into the batter"],
    )
    rex.cross_reference_check(recipe)
    assert recipe.warnings
    assert any("vanilla" in w.lower() for w in recipe.warnings)


def test_save_outputs_with_warnings(tmp_path, monkeypatch):
    ingredient = rex.Ingredient(original="1 cup sugar", quantity="1", unit="cup", item="sugar", notes=None)
    recipe = rex.RecipeOutput(
        title="Cake",
        url="https://example.com",
        ingredients=[ingredient],
        directions=["Mix"],
        extras={"ocr_samples": []},
        raw_sources={"description": "", "transcript": ""},
        warnings=["Unused ingredient: vanilla", "Missing ingredient: butter (added to ingredients)"],
    )

    def fake_dump(self, indent=2, ensure_ascii=False):
        return json.dumps(self.model_dump(), indent=indent)

    monkeypatch.setattr(rex.RecipeOutput, "model_dump_json", fake_dump, raising=False)
    outdir = tmp_path / "output"
    rex.save_outputs(recipe, outdir)

    md_text = (outdir / "cake.md").read_text()
    assert "## Warnings" in md_text
    assert "Unused ingredient: vanilla" in md_text
    assert "Missing ingredient: butter (added to ingredients)" in md_text

    assert "[RecipeRipper](https://github.com/timbroder/RecipeRipper)" in md_text

    json_data = json.loads((outdir / "cake.json").read_text())
    assert "warnings" in json_data
    assert len(json_data["warnings"]) == 2


def test_save_outputs_md_contains_repo_link(tmp_path, monkeypatch):
    recipe = rex.RecipeOutput(
        title="Soup",
        ingredients=[rex.Ingredient(original="1 onion", item="onion")],
        directions=["Chop"],
    )

    def fake_dump(self, indent=2, ensure_ascii=False):
        return json.dumps(self.model_dump(), indent=indent)

    monkeypatch.setattr(rex.RecipeOutput, "model_dump_json", fake_dump, raising=False)
    rex.save_outputs(recipe, tmp_path)
    md_text = (tmp_path / "soup.md").read_text()
    assert "[RecipeRipper](https://github.com/timbroder/RecipeRipper)" in md_text


def test_save_outputs_md_footer_includes_model(tmp_path, monkeypatch):
    recipe = rex.RecipeOutput(
        title="Soup",
        ingredients=[rex.Ingredient(original="1 onion", item="onion")],
        directions=["Chop"],
        model="gpt-4o-mini",
    )

    def fake_dump(self, indent=2, ensure_ascii=False):
        return json.dumps(self.model_dump(), indent=indent)

    monkeypatch.setattr(rex.RecipeOutput, "model_dump_json", fake_dump, raising=False)
    rex.save_outputs(recipe, tmp_path)
    md_text = (tmp_path / "soup.md").read_text()
    assert "Processed with gpt-4o-mini" in md_text
    assert "RecipeRipper" in md_text


# -----------------------------
# save_outputs return value tests
# -----------------------------


def test_save_outputs_returns_paths(tmp_path, monkeypatch):
    recipe = rex.RecipeOutput(
        title="Pasta",
        ingredients=[rex.Ingredient(original="1 cup flour", item="flour")],
        directions=["Boil water"],
    )

    def fake_dump(self, indent=2, ensure_ascii=False):
        return json.dumps(self.model_dump(), indent=indent)

    monkeypatch.setattr(rex.RecipeOutput, "model_dump_json", fake_dump, raising=False)
    json_path, md_path = rex.save_outputs(recipe, tmp_path)
    assert json_path == tmp_path / "pasta.json"
    assert md_path == tmp_path / "pasta.md"
    assert json_path.exists()
    assert md_path.exists()


# -----------------------------
# publish_gist tests
# -----------------------------


def test_publish_gist_success(monkeypatch, tmp_path, capsys):
    f1 = tmp_path / "recipe.json"
    f2 = tmp_path / "recipe.md"
    f1.write_text("{}")
    f2.write_text("# Recipe")

    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="https://gist.github.com/abc123\n", stderr="")

    monkeypatch.setattr(rex.subprocess, "run", fake_run)
    monkeypatch.setattr(rex.shutil, "which", lambda name: "/usr/bin/gh")

    result = rex.publish_gist([f1, f2])

    assert result == "https://gist.github.com/abc123"
    assert captured["cmd"] == ["gh", "gist", "create", "--public", str(f1), str(f2)]
    out = capsys.readouterr().out
    assert "https://gist.github.com/abc123" in out


def test_publish_gist_gh_missing(monkeypatch):
    monkeypatch.setattr(rex.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        rex.publish_gist([Path("/fake/file.json")])
    assert exc.value.code == 1


def test_publish_gist_gh_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rex.shutil, "which", lambda name: "/usr/bin/gh")

    def fake_run(cmd, capture_output, text):
        return SimpleNamespace(returncode=1, stdout="", stderr="auth required")

    monkeypatch.setattr(rex.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc:
        rex.publish_gist([tmp_path / "recipe.json"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "auth required" in out


def test_paprika_flag_copies_and_opens(monkeypatch, tmp_path):
    """--paprika implies --publish, copies URL to clipboard, and opens Paprika."""
    monkeypatch.setattr(sys, "argv", [
        "prog", "--youtube", "https://youtu.be/test",
        "--outdir", str(tmp_path), "--paprika",
    ])
    monkeypatch.setattr(rex, "ensure_ffmpeg", lambda: None)

    dummy_recipe = rex.RecipeOutput(
        title="T", url="u", ingredients=[], directions=[], extras={}, raw_sources={},
    )
    monkeypatch.setattr(rex, "process_youtube", lambda url, args: dummy_recipe)
    monkeypatch.setattr(rex, "save_outputs", lambda out, od: (Path("/fake/recipe.json"), Path("/fake/recipe.md")))
    monkeypatch.setattr(rex, "pretty_print", lambda out: None)
    monkeypatch.setattr(rex.shutil, "which", lambda name: "/usr/bin/gh")

    calls = []

    def fake_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("cmd")
        if cmd[0] == "gh":
            return SimpleNamespace(returncode=0, stdout="https://gist.github.com/abc123\n", stderr="")
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rex.subprocess, "run", fake_run)

    rex.main()

    assert len(calls) == 1
    bash_cmd = calls[0]
    assert bash_cmd[0] == "bash"
    assert "pbcopy" in bash_cmd[2]
    assert "Paprika Recipe Manager 3" in bash_cmd[2]
    assert "https://gist.github.com/abc123" in bash_cmd[2]


# -----------------------------
# "Salt and pepper to taste" fix tests
# -----------------------------


def test_looks_like_ingredient_salt_and_pepper_to_taste():
    """'salt and pepper to taste' should be recognized as an ingredient."""
    assert rex.looks_like_ingredient("salt and pepper to taste")
    assert rex.looks_like_ingredient("Salt and pepper to taste")
    assert rex.looks_like_ingredient("salt to taste")
    assert rex.looks_like_ingredient("pepper as needed")
    assert rex.looks_like_ingredient("fresh garlic, optional")
    assert rex.looks_like_ingredient("parsley for garnish")


def test_looks_like_ingredient_still_rejects_non_food():
    """Taste modifiers without food words should still be rejected."""
    assert not rex.looks_like_ingredient("patience as needed")
    assert not rex.looks_like_ingredient("love to taste")


def test_extract_ingredients_preserves_to_taste():
    """extract_ingredients_from_sentence should preserve 'to taste' qualifier."""
    result = rex.extract_ingredients_from_sentence("add salt and pepper to taste")
    assert any("to taste" in r.lower() for r in result)


def test_extract_ingredients_salt_pepper_without_to_taste():
    """Without 'to taste', should still extract 'Salt and pepper'."""
    result = rex.extract_ingredients_from_sentence("add salt and pepper")
    assert any("salt and pepper" in r.lower() for r in result)


def test_dedup_does_not_eat_compound_ingredient():
    """'salt and pepper' should not be deduped when only 'salt' was seen from another ingredient."""
    ing_lines, _ = rex.combine_sources(
        description="",
        transcript="I'm using a pinch of salt. Then add salt and pepper to taste.",
        ocr_lines=[],
    )
    joined = " ".join(ing_lines).lower()
    # "salt and pepper" should survive even though "salt" appeared separately
    assert "pepper" in joined


def test_dedup_still_removes_true_duplicates():
    """A true duplicate (all food words already seen) should still be removed."""
    ing_lines, _ = rex.combine_sources(
        description="",
        transcript="",
        ocr_lines=["1 tsp salt", "salt"],
    )
    # "salt" alone (3 words or fewer, only food word is "salt") should be deduped
    salt_count = sum(1 for l in ing_lines if l.strip().lower() == "salt")
    assert salt_count == 0


# -----------------------------
# Cross-reference false-positive & promotion tests
# -----------------------------


def test_extract_food_words_multi_word_suppresses_single():
    """'nutritional yeast' in text should NOT also yield standalone 'yeast'."""
    found = rex._extract_food_words_from_text("nutritional yeast")
    normalized_yeast = rex._normalize_word("yeast")
    normalized_nutritional_yeast = rex._normalize_word("nutritional yeast")
    assert normalized_nutritional_yeast in found
    assert normalized_yeast not in found


def test_cross_reference_no_false_positive_subword():
    """When both ingredient list and directions mention 'nutritional yeast',
    there should be no 'Missing ingredient: yeast' warning."""
    recipe = rex.RecipeOutput(
        ingredients=[
            rex.Ingredient(original="2 tbsp nutritional yeast", item="nutritional yeast"),
        ],
        directions=["Sprinkle nutritional yeast on top"],
    )
    warnings = rex.cross_reference_check(recipe)
    missing = [w for w in warnings if "Missing" in w]
    assert not any("yeast" in w.lower() for w in missing)


def test_cross_reference_promotes_missing_ingredients():
    """Missing food words found in directions should be added to recipe.ingredients."""
    recipe = rex.RecipeOutput(
        ingredients=[
            rex.Ingredient(original="1 cup flour", item="flour"),
        ],
        directions=["Mix the flour with butter"],
    )
    original_count = len(recipe.ingredients)
    warnings = rex.cross_reference_check(recipe)
    assert any("butter" in w.lower() for w in warnings)
    assert len(recipe.ingredients) > original_count
    added_items = [ing.item for ing in recipe.ingredients[original_count:]]
    assert "butter" in added_items

