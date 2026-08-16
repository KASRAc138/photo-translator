"""Font selection and complex-script text layout.

Persian and Arabic output is where naive implementations give themselves away:
the letters come out disconnected and in reverse order. There are two ways to
get it right, and which one is available depends on how Pillow was built.

**Raqm (preferred).** Pillow compiled with libraqm delegates to HarfBuzz, which
does real contextual shaping and bidi reordering from raw Unicode. Text is
passed through untouched with ``direction="rtl"``. This is higher quality --
it handles ligatures, kashida and mixed-direction runs that the fallback
mangles -- and it is what ``features.check("raqm")`` is testing for.

**arabic-reshaper + python-bidi (fallback).** Without Raqm, the text must be
pre-converted to Arabic *presentation forms* (U+FE70-U+FEFF) and manually
reversed. This is the trap that produced the tofu boxes during development:
a font can contain Arabic (U+0600-U+06FF) and still lack every presentation
form, so ``DejaVuSans`` renders raw Persian fine and reshaped Persian as
solid squares. Font coverage therefore has to be checked against the codepoints
that will *actually be drawn*, not against the source string.

:func:`pick_font` does exactly that, which is why it needs to know the target
language and the layout engine before it can choose.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

from PIL import ImageFont, features

from .config import is_rtl, resource_dir

log = logging.getLogger("pt.textshape")

# Representative strings per script. Coverage is tested against these.
_SAMPLES = {
    "arabic": "ابپتجچزسشصعکگلمنهی",   # Persian-specific letters included
    "hebrew": "אבגדהוזחטיכלמנסעפצקרשת",
    "cyrillic": "абвгдежзийклмнопрстуфхцчшщъыьэюя",
    "greek": "αβγδεζηθικλμνξοπρστυφχψω",
    "latin": "abcdefghijklmnopqrstuvwxyzäöüßàéîñ",
}

_LANG_SCRIPT = {
    "fa": "arabic", "ar": "arabic", "ur": "arabic", "ps": "arabic",
    "sd": "arabic", "ku": "arabic", "dv": "arabic",
    "he": "hebrew", "iw": "hebrew", "yi": "hebrew",
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic", "sr": "cyrillic",
    "el": "greek",
}

# Best first. The Windows fonts lead because that is the target platform and
# arial/tahoma carry full Arabic presentation forms, which most Linux fonts
# do not.
_PREFERRED = [
    "Vazirmatn-Regular.ttf", "NotoNaskhArabic-Regular.ttf", "NotoSansArabic-Regular.ttf",
    "arial.ttf", "arialuni.ttf", "tahoma.ttf", "segoeui.ttf", "calibri.ttf",
    "NotoSans-Regular.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
    "FreeSerif.ttf", "Carlito-Regular.ttf",
]

_FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
]


@functools.lru_cache(maxsize=1)
def has_raqm() -> bool:
    """True when Pillow can shape complex scripts itself.

    Frozen builds often lose this: PyInstaller collects the Pillow wheel but
    not ``libraqm``/``fribidi``/``harfbuzz`` unless told to. ``build_portable.py``
    checks for it and warns, because losing it silently downgrades RTL output
    to the fallback path without any error.
    """
    try:
        return bool(features.check("raqm"))
    except Exception:
        return False


def script_for(lang: str) -> str:
    return _LANG_SCRIPT.get((lang or "").split("-")[0].split("_")[0].lower(), "latin")


def _drawable(font: ImageFont.FreeTypeFont, text: str) -> float:
    """Fraction of ``text`` the font has real glyphs for.

    Compares each glyph's raster against U+FFFF's, which no font defines, so
    both fall back to .notdef and compare equal when the glyph is missing.
    """
    try:
        notdef = font.getmask("\uffff").getbbox()
    except Exception:
        return 0.0
    hits = total = 0
    for ch in set(text):
        if ch.isspace():
            continue
        total += 1
        try:
            if font.getmask(ch).getbbox() != notdef:
                hits += 1
        except Exception:
            pass
    return hits / total if total else 0.0


def coverage(font_path: str, lang: str, raqm: bool | None = None) -> float:
    """Score a font for ``lang``, testing the codepoints that will be drawn.

    Without Raqm the fallback reshaper emits presentation forms, so those are
    what get tested -- checking the base letters instead is precisely the
    mistake that lets a tofu-rendering font look acceptable.
    """
    raqm = has_raqm() if raqm is None else raqm
    sample = _SAMPLES.get(script_for(lang), _SAMPLES["latin"])
    if is_rtl(lang) and not raqm:
        try:
            import arabic_reshaper
            sample = arabic_reshaper.reshape(sample)
        except ImportError:
            pass
    try:
        font = ImageFont.truetype(font_path, 24)
    except Exception:
        return 0.0
    return _drawable(font, sample)


def _candidates() -> list[Path]:
    """Bundled fonts first, then preferred system fonts, then anything."""
    found: list[Path] = []
    bundled = resource_dir("fonts")
    if bundled is not None:
        for name in _PREFERRED:
            hit = bundled / name
            if hit.is_file():
                found.append(hit)
        found.extend(sorted(p for p in bundled.glob("*.ttf")))
        found.extend(sorted(p for p in bundled.glob("*.otf")))

    for directory in _FONT_DIRS:
        if not directory.is_dir():
            continue
        for name in _PREFERRED:
            hit = directory / name
            if hit.is_file():
                found.append(hit)
        try:
            found.extend(sorted(directory.rglob("*.ttf"))[:120])
        except OSError:
            pass

    seen, unique = set(), []
    for path in found:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


@functools.lru_cache(maxsize=8)
def pick_font(lang: str, explicit: str | None = None) -> str:
    """Best available font for ``lang``. Raises if nothing can draw the script.

    Failing loudly beats rendering a page of squares: tofu output looks like a
    shaping bug and sends you debugging the wrong module entirely.
    """
    if explicit and Path(explicit).is_file():
        score = coverage(explicit, lang)
        if score < 0.9:
            log.warning("PT_FONT covers only %.0f%% of the %s script; using it anyway",
                        score * 100, script_for(lang))
        return explicit

    best, best_score = None, 0.0
    for path in _candidates():
        score = coverage(str(path), lang)
        if score > best_score:
            best, best_score = str(path), score
        if best_score >= 0.999:
            break

    if best is None or best_score < 0.5:
        raise RuntimeError(
            f"No installed font can render {lang!r} ({script_for(lang)} script"
            + ("" if has_raqm() else ", presentation forms")
            + "). Put a suitable .ttf in the app's fonts/ folder "
            "(Vazirmatn or Noto Naskh Arabic for Persian) or set PT_FONT."
        )
    if best_score < 0.999:
        log.warning("Best font %s covers %.0f%% of %s", Path(best).name, best_score * 100, lang)
    return best


def prepare(text: str, lang: str) -> str:
    """Return the string to hand to Pillow, given the active layout engine.

    With Raqm this is the identity -- HarfBuzz wants the logical order and
    does the reordering itself. Pre-reshaping *and* letting Raqm reorder would
    reverse the text twice and put it back in the wrong order.
    """
    if not text or not is_rtl(lang) or has_raqm():
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        log.warning(
            "RTL target, no Raqm, and arabic-reshaper/python-bidi missing -- "
            "text will render as disconnected letters. "
            "pip install arabic-reshaper python-bidi"
        )
        return text


def draw_kwargs(lang: str) -> dict:
    """Extra kwargs for ``ImageDraw.text``/``textlength``.

    ``direction`` is only legal when Raqm is present; passing it otherwise
    raises, which is why this is centralised rather than inlined at the call
    sites.
    """
    if is_rtl(lang) and has_raqm():
        return {"direction": "rtl"}
    return {}


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load at ``size`` with the layout engine that matches :func:`prepare`.

    The two must agree or Persian comes out reversed. ``prepare`` reshapes and
    bidi-reorders only when Raqm is absent; if the font were then loaded with a
    layout engine that *does* reorder, the text would be reversed twice and
    read backwards while still looking correctly joined.

    So the engine is pinned explicitly in both directions -- RAQM when we are
    passing raw Unicode, BASIC when we have already reordered by hand -- rather
    than leaving Pillow to choose. Leaving it to choose is what made this
    subtle: the default picks Raqm when available, which silently contradicts
    the fallback path.
    """
    engine = ImageFont.Layout.RAQM if has_raqm() else ImageFont.Layout.BASIC
    try:
        return ImageFont.truetype(path, size, layout_engine=engine)
    except Exception:
        return ImageFont.truetype(path, size)
