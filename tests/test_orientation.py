"""Regression tests for per-box text orientation.

One page routinely carries body text at 0 degrees and a figure caption at 90 or
270. The detector reports the caption's *text* correctly but hands back a tall,
narrow quad with ``angle == 0.0``, so a renderer that trusts the geometry
typesets the translation into a 40px column.

The subtle failure this file exists to prevent is the one that shipped first:
comparing the probe's confidence against the *detector's* confidence. RapidOCR
rotates internally before recognising, so a vertical caption arrives with the
right string at 0.99 confidence attached to horizontal geometry. Seeded as a
baseline, that 0.99 is unbeatable and the probe concludes every box is upright
-- passing every test that only checks horizontal pages.

``test_probe_baseline_is_not_detector_confidence`` pins that down specifically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pt import Config, Pipeline  # noqa: E402
from pt import orient  # noqa: E402
from pt.geometry import TextBox  # noqa: E402
from pt.imageio import load  # noqa: E402
from pt.ocr import RapidOcrEngine  # noqa: E402
from pt.textshape import load_font, pick_font  # noqa: E402
from pt.translate import IdentityTranslator  # noqa: E402

PAPER = (250, 248, 242)
INK = (20, 20, 20)

HORIZONTAL = ["Die Zubereitung des Tees", "erfordert heisses Wasser"]
DOWNWARD = "Quelle Archiv"          # rotated clockwise: reads top-to-bottom
UPWARD = "Abbildung 3 Teekanne"     # rotated anti-clockwise: reads bottom-to-top


def _vertical_tile(text: str, font, clockwise: bool) -> Image.Image:
    strip = Image.new("RGBA", (int(font.getlength(text)) + 12, 62), (0, 0, 0, 0))
    ImageDraw.Draw(strip).text((6, 4), text, font=font, fill=(*INK, 255))
    return strip.rotate(-90 if clockwise else 90, expand=True)


@pytest.fixture(scope="module")
def mixed_page(tmp_path_factory) -> Path:
    """A page with horizontal, clockwise and anti-clockwise text at once."""
    font = load_font(pick_font("de"), 40)
    img = Image.new("RGB", (1000, 700), PAPER)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(HORIZONTAL):
        draw.text((60, 60 + i * 70), line, font=font, fill=INK)

    down = _vertical_tile(DOWNWARD, font, clockwise=True)
    img.paste(down, (60, 250), down)
    up = _vertical_tile(UPWARD, font, clockwise=False)
    img.paste(up, (870, 230), up)

    path = tmp_path_factory.mktemp("orient") / "mixed.png"
    img.save(path)
    return path


@pytest.fixture(scope="module")
def detected(mixed_page):
    image = load(mixed_page)
    engine = RapidOcrEngine(Config())
    return image, engine.detect(image)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_text_size_swaps_for_vertical_runs():
    """A vertical box's usable length is its quad height, not its width."""
    box = TextBox([(0, 0), (40, 0), (40, 300), (0, 300)], "x")
    assert box.text_size == (40.0, 300.0)      # horizontal reading
    box.orientation = 90
    assert box.text_size == (300.0, 40.0)      # rotated: length and thickness swap
    assert box.is_vertical


def test_render_angle_composes_skew_and_orientation():
    """Page tilt and reading rotation are independent and must add."""
    import math
    theta = math.radians(5.0)
    dx, dy = 200 * math.cos(theta), 200 * math.sin(theta)
    box = TextBox([(0, 0), (dx, dy), (dx - 20, 40 + dy), (-20, 40)], "x")

    flat = box.render_angle_deg
    box.orientation = 90
    assert box.render_angle_deg == pytest.approx(flat - 90, abs=0.01)


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def test_candidates_depend_on_aspect():
    tall = np.zeros((300, 40, 3), dtype=np.uint8)
    wide = np.zeros((40, 300, 3), dtype=np.uint8)
    assert orient.candidates_for(tall, 0.99) == [90, 270, 0]
    assert orient.candidates_for(wide, 0.99) == [0]          # confident: no probe
    assert orient.candidates_for(wide, 0.20) == [0, 180]     # unsure: check upside down


def test_detects_both_vertical_directions(detected):
    """Clockwise and anti-clockwise captions must be told apart."""
    _, boxes = detected
    found = {b.text.strip(): b.orientation for b in boxes}

    downward = [o for t, o in found.items() if "Quelle" in t or "Archiv" in t]
    upward = [o for t, o in found.items() if "Teekanne" in t or "bbildung" in t]

    assert downward, f"clockwise caption not detected at all: {list(found)}"
    assert upward, f"anti-clockwise caption not detected at all: {list(found)}"
    assert downward[0] in (90, 270)
    assert upward[0] in (90, 270)
    assert downward[0] != upward[0], (
        "both vertical captions got the same orientation -- the probe is not "
        "distinguishing reading direction"
    )


def test_horizontal_text_left_alone(detected):
    """The probe must not rotate ordinary body text."""
    _, boxes = detected
    for box in boxes:
        if any(word in box.text for word in ("Zubereitung", "erfordert")):
            assert box.orientation == 0, f"{box.text!r} was wrongly rotated"


def test_probe_baseline_is_not_detector_confidence(detected):
    """The bug that shipped first: an unbeatable baseline.

    A vertical caption arrives from the detector with correct text at ~0.99.
    Probing at 0 degrees on the same crop scores ~0.00. If the two are ever
    compared, orientation detection silently stops working.
    """
    image, boxes = detected
    engine = RapidOcrEngine(Config())._load()

    vertical = [b for b in boxes if b.is_vertical]
    assert vertical, "no vertical box to test against"

    box = vertical[0]
    crop = orient.warp_quad(image.bgr, box)
    upright_text, upright_conf = orient.recognise(engine, crop)
    rotated_text, rotated_conf = orient.recognise(
        engine, orient._rotate(crop, box.orientation)
    )

    assert rotated_conf > upright_conf, (
        "the crop reads no better rotated than upright; fixture is not vertical"
    )
    assert rotated_conf > 0.5 and rotated_text.strip()
    # The separation is what makes argmax sound. If this narrows, revisit the
    # near-tie margin in orient.resolve.
    assert rotated_conf - upright_conf > 0.3


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_vertical_text_rendered_vertically(mixed_page, tmp_path):
    """The rendered caption must occupy a tall column, not a wide strip.

    Measured from the ink itself: a correctly rotated caption is far taller
    than it is wide. Before the fix, the same region held two enormous
    horizontal letters, which is wider than tall.
    """
    cfg = Config()
    cfg.output_dir = tmp_path
    cfg.source_lang = cfg.target_lang = "de"
    cfg.overwrite = True
    cfg.erase_mode = "fill"

    result = Pipeline(cfg, translator=IdentityTranslator()).process_image(mixed_page)
    assert result.ok, result.error

    with Image.open(result.output) as img:
        arr = np.asarray(img.convert("L"))

    # Left column strip, where the clockwise caption lives.
    region = arr[200:560, 40:200]
    ink = region < 128
    assert ink.sum() > 300, "no text rendered in the vertical caption region"

    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    height = rows.max() - rows.min()
    width = cols.max() - cols.min()
    assert height > width * 1.5, (
        f"caption rendered {width}x{height} -- too wide to be vertical text"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
