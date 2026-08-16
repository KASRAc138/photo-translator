"""Configuration: env vars, paths, and where things cache.

Every knob has a working default. The app must run with zero configuration on
a machine that has never seen it, because that is the whole point of shipping
an .exe.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def app_dir() -> Path:
    """Directory the app lives in -- next to the .exe when frozen.

    PyInstaller unpacks a one-file build into a temp dir and points
    ``__file__`` there, so ``sys.executable`` is the only thing that tells you
    where the user actually put the program. ``input/`` and ``output/`` must
    appear next to the exe the user double-clicked, not inside a temp dir that
    is deleted on exit.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_dir(name: str) -> Path | None:
    """Locate a bundled data folder, frozen or not.

    PyInstaller unpacks ``--add-data`` payloads into ``sys._MEIPASS``, which is
    a temp directory, *not* where ``sys.executable`` lives. ``app_dir()``
    deliberately points at the exe so ``input/`` and ``output/`` land next to
    it -- which means bundled resources need a different lookup entirely.

    Getting this wrong is quiet: the fonts ship inside the exe, are never
    found, and font selection silently falls back to whatever the host has.
    On a Linux test build that meant a Japanese font was chosen for Latin
    text; on a machine with no Arabic font it would mean Persian could not be
    rendered at all despite Vazirmatn being right there in the archive.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    candidates = []
    if meipass:
        candidates += [Path(meipass) / name, Path(meipass) / "pt" / name]
    candidates.append(app_dir() / name)
    candidates.append(Path(__file__).resolve().parent.parent / name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def data_dir() -> Path:
    """Where models and language packs cache. Beside the app, so it stays portable."""
    override = os.environ.get("PT_DATA_DIR")
    base = Path(override).expanduser() if override else app_dir() / ".pt_data"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # -- languages --------------------------------------------------------
    # "auto" asks the OCR result to be language-detected; anything else is an
    # ISO 639-1 code passed straight through.
    source_lang: str = field(default_factory=lambda: os.environ.get("PT_FROM", "auto"))
    target_lang: str = field(default_factory=lambda: os.environ.get("PT_TO", "en"))

    # -- engines ----------------------------------------------------------
    # "auto" picks Unlimited-OCR when a CUDA device is present, else RapidOCR.
    ocr_engine: str = field(default_factory=lambda: os.environ.get("PT_OCR", "auto"))
    translate_engine: str = field(default_factory=lambda: os.environ.get("PT_TRANSLATE", "argos"))

    # -- folders ----------------------------------------------------------
    input_dir: Path = field(default_factory=lambda: Path(os.environ.get("PT_INPUT", app_dir() / "input")))
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("PT_OUTPUT", app_dir() / "output")))

    # -- OCR tuning -------------------------------------------------------
    min_confidence: float = field(default_factory=lambda: _env_float("PT_MIN_CONF", 0.5))
    # Boxes thinner than this many pixels are page noise, not text.
    min_box_height: float = field(default_factory=lambda: _env_float("PT_MIN_BOX_H", 8.0))
    # Probe each box for its reading rotation. Costs one extra recognition pass
    # per ambiguous box; without it, vertical text is typeset into a sliver.
    detect_orientation: bool = field(default_factory=lambda: not _env_flag("PT_NO_ORIENT"))

    # -- rendering --------------------------------------------------------
    # How the original text is removed before the translation is drawn.
    #   "inpaint" -- OpenCV Telea; best on photos and textured paper
    #   "fill"    -- flat median background colour; best on clean scans
    #   "none"    -- draw over the top; for debugging box placement
    erase_mode: str = field(default_factory=lambda: os.environ.get("PT_ERASE", "inpaint"))
    font_path: str | None = field(default_factory=lambda: os.environ.get("PT_FONT"))
    # Translations are usually longer than the source. Allow the text to
    # shrink to this fraction of the box height before it is allowed to
    # overflow rather than becoming unreadable.
    min_font_scale: float = field(default_factory=lambda: _env_float("PT_MIN_FONT_SCALE", 0.45))
    draw_debug_boxes: bool = field(default_factory=lambda: _env_flag("PT_DEBUG_BOXES"))

    # -- behaviour --------------------------------------------------------
    jpeg_quality: int = field(default_factory=lambda: _env_int("PT_QUALITY", 95))
    overwrite: bool = field(default_factory=lambda: _env_flag("PT_OVERWRITE", False))
    verbose: bool = field(default_factory=lambda: _env_flag("PT_VERBOSE", True))
    offline: bool = field(default_factory=lambda: _env_flag("PT_OFFLINE", False))

    def ensure_folders(self) -> tuple[Path, Path]:
        """Create ``input/`` and ``output/``. Called on every run, not just the first."""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.input_dir, self.output_dir

    def describe(self) -> str:
        return (
            f"  source     {self.source_lang}\n"
            f"  target     {self.target_lang}\n"
            f"  ocr        {self.ocr_engine}\n"
            f"  translate  {self.translate_engine}\n"
            f"  erase      {self.erase_mode}\n"
            f"  input      {self.input_dir}\n"
            f"  output     {self.output_dir}"
        )


# Languages written right-to-left. These need shaping and bidi reordering
# before they are drawn, or every word comes out as disconnected glyphs in
# reverse order -- the single most common failure in naive implementations.
RTL_LANGS = {"fa", "ar", "he", "iw", "ur", "ps", "sd", "ku", "yi", "dv"}


def is_rtl(lang_code: str) -> bool:
    return (lang_code or "").split("-")[0].split("_")[0].lower() in RTL_LANGS
