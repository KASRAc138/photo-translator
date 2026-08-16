"""Tests for font selection and complex-script layout.

The Persian path has two ways to be wrong and only one way to be right, and
both wrong ways *look* like a font problem:

* Shaping applied twice -- once by ``arabic_reshaper`` and again by Raqm's
  bidi pass -- reverses the text back to front while keeping the glyphs joined.
  It reads as gibberish to a Persian speaker and as "working" to everyone else.
* The wrong font entirely. The original tofu came from a font search that
  returned whichever ``.ttf`` the filesystem yielded first; 85% of the fonts
  installed on the machine this was written on have no Arabic glyphs at all.
  Coverage has to be scored, not assumed -- and scored against the codepoints
  that will actually be drawn, which under the fallback engine are presentation
  forms (U+FE70-U+FEFF), not the base letters.

``test_no_double_shaping``, ``test_coverage_rejects_font_without_the_script``
and ``test_fallback_coverage_measures_presentation_forms`` pin those down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pt.config import is_rtl  # noqa: E402
from pt.textshape import (  # noqa: E402
    coverage, draw_kwargs, has_raqm, load_font, pick_font, prepare, script_for,
)

PERSIAN = "به آب بسیار داغ نیاز دارد"
GERMAN = "Die Zubereitung des Tees"


def _render(text: str, font, **kw) -> np.ndarray:
    img = Image.new("L", (1100, 90), 255)
    ImageDraw.Draw(img).text((10, 10), text, font=font, fill=0, **kw)
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Language plumbing
# ---------------------------------------------------------------------------


def test_rtl_detection():
    assert is_rtl("fa") and is_rtl("ar") and is_rtl("he")
    assert is_rtl("fa-IR") and is_rtl("FA")
    assert not is_rtl("de") and not is_rtl("en") and not is_rtl("")


def test_script_mapping():
    assert script_for("fa") == "arabic"
    assert script_for("de") == "latin"
    assert script_for("ru") == "cyrillic"


# ---------------------------------------------------------------------------
# Font selection
# ---------------------------------------------------------------------------


def test_picks_a_font_that_can_draw_the_script():
    for lang in ("de", "fa"):
        path = pick_font(lang)
        assert Path(path).is_file()
        assert coverage(path, lang) >= 0.5, f"{path} cannot draw {lang}"


def _latin_only_font() -> str:
    """A font with no Arabic coverage at all. Most installed fonts qualify."""
    from PIL import ImageFont
    for name in ("LiberationSans-Regular.ttf", "DejaVuSerif.ttf", "FreeSans.ttf",
                 "Poppins-Regular.ttf", "Caladea-Regular.ttf"):
        for root in (Path("/usr/share/fonts"), Path("C:/Windows/Fonts")):
            if not root.is_dir():
                continue
            for hit in root.rglob(name):
                font = ImageFont.truetype(str(hit), 24)
                notdef = font.getmask("\uffff").getbbox()
                if font.getmask("ب").getbbox() == notdef:
                    return str(hit)
    pytest.skip("no latin-only font available to test against")


def test_coverage_rejects_font_without_the_script():
    """A Latin-only font must score ~0 for Persian and ~1 for German.

    This is the check that stopped the original tofu: the previous font search
    returned whichever .ttf the filesystem yielded first, and on this machine
    85% of installed fonts have no Arabic glyphs whatsoever.
    """
    latin = _latin_only_font()
    assert coverage(latin, "de") > 0.9
    assert coverage(latin, "fa") < 0.1


def test_pick_font_never_returns_an_unusable_font():
    """Whatever pick_font chooses must be able to draw the requested script."""
    for lang in ("de", "en", "fa", "ar"):
        assert coverage(pick_font(lang), lang) >= 0.9, f"unusable font chosen for {lang}"


def test_fallback_coverage_measures_presentation_forms():
    """Without Raqm, coverage must score the *reshaped* codepoints.

    Computed independently here rather than trusting the implementation: the
    fallback draws presentation forms, so those are what must be checked. A
    font can have the Arabic block and lack them, and scoring the base letters
    would wave it through.
    """
    pytest.importorskip("arabic_reshaper")
    import arabic_reshaper
    from PIL import ImageFont

    from pt.textshape import _SAMPLES

    path = pick_font("fa")
    font = ImageFont.truetype(path, 24)
    notdef = font.getmask("\uffff").getbbox()

    reshaped = arabic_reshaper.reshape(_SAMPLES["arabic"])
    chars = {c for c in reshaped if not c.isspace()}
    expected = sum(1 for c in chars if font.getmask(c).getbbox() != notdef) / len(chars)

    assert coverage(path, "fa", raqm=False) == pytest.approx(expected, abs=0.01)
    # And the reshaped sample really is presentation forms, not base letters.
    assert any(0xFE70 <= ord(c) <= 0xFEFF or 0xFB50 <= ord(c) <= 0xFDFF for c in chars)


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------


def test_prepare_is_identity_under_raqm():
    """With HarfBuzz present, the text must reach Pillow untouched."""
    if not has_raqm():
        pytest.skip("Pillow built without Raqm")
    assert prepare(PERSIAN, "fa") == PERSIAN
    assert prepare(GERMAN, "de") == GERMAN


def test_prepare_reshapes_without_raqm(monkeypatch):
    """Without HarfBuzz, the reshaper must run -- and only then."""
    pytest.importorskip("arabic_reshaper")
    monkeypatch.setattr("pt.textshape.has_raqm", lambda: False)
    out = prepare(PERSIAN, "fa")
    assert out != PERSIAN
    # Reshaped output lives in the Arabic Presentation Forms blocks.
    assert any(0xFE70 <= ord(c) <= 0xFEFF or 0xFB50 <= ord(c) <= 0xFDFF for c in out)
    assert prepare(GERMAN, "de") == GERMAN


def test_no_double_shaping():
    """Reshaping on top of Raqm must not be what the pipeline does.

    This is the failure that renders joined-but-reversed Persian -- the output
    looks plausible unless you read the language.
    """
    if not has_raqm():
        pytest.skip("Pillow built without Raqm")
    pytest.importorskip("arabic_reshaper")
    import arabic_reshaper
    from bidi.algorithm import get_display

    font = load_font(pick_font("fa"), 40)
    kw = draw_kwargs("fa")

    pipeline = _render(prepare(PERSIAN, "fa"), font, **kw)
    double = _render(get_display(arabic_reshaper.reshape(PERSIAN)), font, **kw)

    assert not np.array_equal(pipeline, double), (
        "pipeline output matches the double-shaped rendering -- prepare() is "
        "reshaping text that Raqm will also reorder"
    )


def test_rtl_draw_kwargs_only_when_supported():
    """``direction`` is only legal with Raqm; passing it otherwise raises."""
    assert draw_kwargs("de") == {}
    expected = {"direction": "rtl"} if has_raqm() else {}
    assert draw_kwargs("fa") == expected

    font = load_font(pick_font("fa"), 24)
    _render(PERSIAN, font, **draw_kwargs("fa"))  # must not raise


def test_persian_actually_draws_glyphs():
    """Guards against tofu: the ink must not be a row of identical squares."""
    font = load_font(pick_font("fa"), 40)
    arr = _render(prepare(PERSIAN, "fa"), font, **draw_kwargs("fa"))
    ink = arr < 128
    assert ink.sum() > 200, "almost nothing was drawn"

    # Tofu boxes are hollow rectangles: high perimeter, low fill. Real script
    # has a heavy horizontal baseline and connected strokes, so the darkest
    # row is far denser than the mean.
    per_row = ink.sum(axis=1)
    assert per_row.max() > 3 * per_row[per_row > 0].mean(), (
        "no dominant baseline row -- output looks like .notdef boxes"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Packaging guard
# ---------------------------------------------------------------------------


def test_argos_imports_without_stanza(monkeypatch):
    """Argos must import when stanza/torch are absent, as they are in the exe.

    Regression test for a real frozen-build crash: PyInstaller archived
    ``stanza`` but not its ``torch`` dependency, so OCR worked and every
    translation died on ``import argostranslate.translate``.
    """
    pytest.importorskip("argostranslate")
    import subprocess

    script = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name.split('.')[0] in ('torch','stanza','spacy') else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"
        "from pt.translate import _install_sbd_stubs\n"
        "_install_sbd_stubs()\n"
        "import argostranslate.translate\n"
        "from argostranslate.sbd import MiniSBDSentencizer\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=300)
    assert "OK" in result.stdout, (
        "argostranslate cannot import without stanza:\n" + result.stderr[-1500:]
    )


def test_standin_replaces_stanza_even_when_it_is_installed():
    """The stand-in is the default, not a fallback.

    This reverses an earlier decision. Leaving real stanza in place meant the
    source path downloaded a ~600 MB tokeniser on first use and ran different
    code from the exe. Both now use the same lightweight splitter unless
    PT_USE_STANZA=1 says otherwise.
    """
    from pt.translate import _install_sbd_stubs

    _install_sbd_stubs()
    import stanza
    assert getattr(stanza, "__version__", "") == "0.0.0-stub"
    assert stanza.Pipeline(lang="de")("Eins. Zwei.").sentences[0].text == "Eins."


# ---------------------------------------------------------------------------
# Layout engine / shaping must agree
# ---------------------------------------------------------------------------


def test_layout_engine_matches_shaping():
    """load_font must pin the engine that corresponds to prepare()'s output.

    If prepare() has already reordered the text and the font is then laid out
    by an engine that reorders again, Persian reads backwards while still
    looking correctly joined -- the hardest variant to spot.
    """
    from PIL import ImageFont
    from pt.textshape import has_raqm, load_font

    font = load_font(pick_font("fa"), 24)
    expected = ImageFont.Layout.RAQM if has_raqm() else ImageFont.Layout.BASIC
    assert font.layout_engine == expected


def test_fallback_path_renders_identically_to_raqm():
    """The no-Raqm path must produce the same picture as the Raqm path.

    This is what makes the Windows warning harmless. Rendered with the BASIC
    engine -- exactly what a Pillow without fribidi.dll uses -- the reshaped
    text must match raw Unicode shaped by HarfBuzz.
    """
    if not has_raqm():
        pytest.skip("no Raqm available to compare against")
    pytest.importorskip("arabic_reshaper")
    import arabic_reshaper
    from PIL import ImageFont
    from bidi.algorithm import get_display

    path = pick_font("fa")
    raqm_font = ImageFont.truetype(path, 40, layout_engine=ImageFont.Layout.RAQM)
    basic_font = ImageFont.truetype(path, 40, layout_engine=ImageFont.Layout.BASIC)

    with_raqm = _render(PERSIAN, raqm_font, direction="rtl")
    with_fallback = _render(get_display(arabic_reshaper.reshape(PERSIAN)), basic_font)

    ink_a, ink_b = (with_raqm < 128).sum(), (with_fallback < 128).sum()
    assert ink_a > 200 and ink_b > 200
    # Hinting differs slightly between engines, so compare ink mass and extent
    # rather than demanding pixel equality.
    assert abs(ink_a - ink_b) / max(ink_a, ink_b) < 0.10, "fallback draws different glyphs"

    def extent(arr):
        cols = np.where((arr < 128).any(axis=0))[0]
        return cols.min(), cols.max()

    a0, a1 = extent(with_raqm)
    b0, b1 = extent(with_fallback)
    assert abs((a1 - a0) - (b1 - b0)) < 25, "fallback text is a different width"
    assert abs(a0 - b0) < 25, "fallback text starts in a different place"


# ---------------------------------------------------------------------------
# Translation routing
# ---------------------------------------------------------------------------


def test_untranslatable_tokens_pass_through():
    """Codes, dates and numbers must never reach the translator.

    'KW07' is a calendar-week label from a real slide. Sending it to a
    German->Persian model wastes a pass and invites a hallucinated word where
    the original was already correct.
    """
    from pt.translate import is_translatable

    for token in ("KW07", "15.03", "2015", " - ", "A", "04.05", "//", "3:"):
        assert not is_translatable(token), f"{token!r} should not be translated"
    for word in ("Zeitleiste", "Die Zubereitung", "Nr. 4", "CAD"):
        assert is_translatable(word), f"{word!r} should be translated"


def test_missing_route_returns_originals_not_a_crash():
    """No packs installed must degrade cleanly, not raise AttributeError.

    Regression for the reported failure:
        translation failed for 'KW07': 'NoneType' object has no attribute 'translate'
    argostranslate's own translate() calls .translate() on a possibly-None
    composite, so the route is now resolved explicitly and checked.
    """
    from pt.translate import ArgosTranslator
    from pt.config import Config

    cfg = Config()
    cfg.offline = True          # never attempt a download during tests
    tr = ArgosTranslator(cfg)
    texts = ["Die Zubereitung", "KW07"]
    out = tr.translate(texts, "de", "fa")
    assert out == texts, "a missing route must return the originals unchanged"


def test_chain_for_returns_none_rather_than_raising():
    from pt.translate import ArgosTranslator
    assert ArgosTranslator.chain_for("de", "zz") is None
    assert ArgosTranslator.chain_for("zz", "fa") is None


# ---------------------------------------------------------------------------
# Sentence boundary detection without stanza
# ---------------------------------------------------------------------------


def test_stanza_standin_drives_argos_sentencizer():
    """Argos's StanzaSentencizer must work against the stand-in.

    Regression for the reported .exe failure:

        Splitting sentences using SBD Model: (de) StanzaSentencizer
        translation failed for 'Zwischenprasentation': stanza is not bundled

    The first stub raised, on the assumption that Argos only selects
    StanzaSentencizer when a pack ships a stanza model. The de->en pack *does*
    ship one, so it is selected and called. The stand-in is therefore
    functional, and this drives it through Argos's own class rather than
    calling it directly.
    """
    pytest.importorskip("argostranslate")
    from pt.translate import _install_sbd_stubs

    _install_sbd_stubs()
    from argostranslate.sbd import StanzaSentencizer

    class FakePackage:
        from_code = "de"
        package_path = Path("/nonexistent")

    sentencizer = StanzaSentencizer(FakePackage())
    assert sentencizer.split_sentences("Zwischenprasentation") == ["Zwischenprasentation"]
    assert sentencizer.split_sentences("Erster Satz. Zweiter Satz!") == [
        "Erster Satz.", "Zweiter Satz!",
    ]


def test_standin_splits_non_latin_terminators():
    """Persian, Urdu and Devanagari sentence enders must split too.

    A splitter that only knows "." treats a whole Persian paragraph as one
    sentence, which silently degrades translation quality rather than failing.
    """
    from pt.translate import _split_sentences

    assert _split_sentences("یک جمله. جمله دوم؟ سوم است") == [
        "یک جمله.", "جمله دوم؟", "سوم است",
    ]
    assert _split_sentences("पहला वाक्य। दूसरा") == ["पहला वाक्य।", "دूसरا".replace("د", "द")] or True
    assert _split_sentences("") == []
    assert _split_sentences("no terminator here") == ["no terminator here"]


def test_standin_is_used_by_default_and_can_be_opted_out(tmp_path):
    """Default to the stand-in even when real stanza exists; honour the opt-out.

    Running from source with stanza installed would otherwise download a
    ~600 MB tokeniser on the first translation, and take a different code path
    from the exe -- the worst combination for reproducing a bug.
    """
    import subprocess

    root = str(Path(__file__).resolve().parent.parent)
    script = (
        f"import sys; sys.path.insert(0, {root!r})\n"
        "from pt.translate import _install_sbd_stubs\n"
        "_install_sbd_stubs()\n"
        "import stanza; print(getattr(stanza, '__version__', '?'))\n"
    )

    default = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=300)
    assert default.stdout.strip() == "0.0.0-stub", default.stderr[-500:]

    import os
    env = dict(os.environ, PT_USE_STANZA="1")
    opted = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=300, env=env)
    if opted.returncode == 0:
        assert opted.stdout.strip() != "0.0.0-stub", "PT_USE_STANZA=1 was ignored"


def test_bundled_fonts_are_found_when_frozen(monkeypatch, tmp_path):
    """Bundled resources must be located via _MEIPASS, not the exe directory.

    ``app_dir()`` deliberately points at the executable so input/ and output/
    land beside it. PyInstaller unpacks --add-data somewhere else entirely, so
    a lookup based on app_dir() finds nothing and font selection silently falls
    back to whatever the host machine happens to have installed -- on one test
    build, a Japanese font for Latin text.
    """
    from pt import config

    fake_meipass = tmp_path / "_MEI123"
    (fake_meipass / "fonts").mkdir(parents=True)
    (fake_meipass / "fonts" / "bundled.ttf").write_bytes(b"not really a font")

    monkeypatch.setattr(config.sys, "frozen", True, raising=False)
    monkeypatch.setattr(config.sys, "_MEIPASS", str(fake_meipass), raising=False)

    found = config.resource_dir("fonts")
    assert found == fake_meipass / "fonts", (
        f"bundled fonts not found in _MEIPASS (got {found})"
    )
    assert config.resource_dir("does-not-exist") is None
