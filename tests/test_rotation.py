"""Regression tests for the rotation bug.

These are written to fail against the *old* shape of the code, not just to
pass against the new one. Specifically:

* ``test_exif_orientations_converge`` fails whenever the load path stops
  normalising orientation, and ``test_naive_load_actually_differs`` proves the
  test has teeth by showing the naive path disagrees on the same four files.
* ``test_no_second_load_path`` fails the moment anyone adds an ``Image.open``
  or ``cv2.imread`` outside ``imageio.py`` -- which is how the bug would come
  back.
* ``test_rotation_sign`` pins the sign of the angle conversion, so a flip
  cannot slip through as "text leans slightly wrong".

Run with:  python -m pytest tests/ -v      (or: python tests/test_rotation.py)
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pt import Config, Pipeline  # noqa: E402
from pt import imageio as pt_imageio  # noqa: E402
from pt.geometry import TextBox, degrees_for_pil  # noqa: E402
from pt.textshape import pick_font  # noqa: E402
from pt.translate import IdentityTranslator  # noqa: E402

ORIENTATION_TAG = 0x0112

# Stored-pixel transform that, when the viewer applies the declared
# orientation, yields the upright original. 5 and 7 are involutions, as are
# 2, 3 and 4; only 6 and 8 need a genuine inverse.
_INVERSE = {
    1: None,
    2: Image.FLIP_LEFT_RIGHT,
    3: Image.ROTATE_180,
    4: Image.FLIP_TOP_BOTTOM,
    5: Image.TRANSPOSE,
    6: Image.ROTATE_90,    # display applies ROTATE_270
    7: Image.TRANSVERSE,
    8: Image.ROTATE_270,   # display applies ROTATE_90
}


def make_page(width: int = 900, height: int = 420) -> Image.Image:
    """A synthetic scan: dark text on off-white paper, asymmetric on both axes.

    Asymmetry is the point. A symmetric test page passes a 180-degree bug.
    """
    img = Image.new("RGB", (width, height), (247, 244, 236))
    draw = ImageDraw.Draw(img)
    font_path = pick_font("de")
    big = ImageFont.truetype(font_path, 46)
    small = ImageFont.truetype(font_path, 30)

    draw.text((40, 50), "Die Zubereitung des Tees", font=big, fill=(20, 20, 24))
    draw.text((40, 150), "erfordert heisses Wasser", font=small, fill=(30, 30, 34))
    # A corner marker so any flip -- not just a rotation -- is detectable.
    draw.rectangle([width - 90, height - 70, width - 40, height - 40], fill=(180, 40, 40))
    return img


def write_with_orientation(img: Image.Image, path: Path, orientation: int) -> Path:
    """Write ``img`` such that a correct reader displays it upright."""
    stored = img if _INVERSE[orientation] is None else img.transpose(_INVERSE[orientation])
    exif = Image.Exif()
    exif[ORIENTATION_TAG] = orientation
    stored.save(path, "JPEG", quality=97, exif=exif)
    return path


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    base = make_page()
    folder = tmp_path_factory.mktemp("orient")
    return base, {o: write_with_orientation(base, folder / f"o{o}.jpg", o)
                  for o in (1, 3, 6, 8)}


# ---------------------------------------------------------------------------
# The bug itself
# ---------------------------------------------------------------------------


def test_exif_orientations_converge(pages):
    """All four files decode to the same upright pixels through imageio.load."""
    _, files = pages
    loaded = {o: pt_imageio.load(p) for o, p in files.items()}

    sizes = {o: img.size for o, img in loaded.items()}
    assert len(set(sizes.values())) == 1, f"sizes disagree after load: {sizes}"

    reference = loaded[1].rgb
    for orientation, img in loaded.items():
        diff = float(np.abs(img.rgb.astype(np.int16) - reference.astype(np.int16)).mean())
        # JPEG round-trips through different stored rotations, so a couple of
        # levels of chroma noise is expected; a wrong rotation scores 40+.
        assert diff < 6.0, f"orientation {orientation} decoded differently (mean diff {diff:.1f})"


def test_naive_load_actually_differs(pages):
    """Proves the previous test is not vacuous.

    The naive path -- plain ``Image.open``, which is what ``cv2.imread`` also
    effectively does -- returns genuinely different pixels for the same four
    files. That difference is the bug.
    """
    _, files = pages
    shapes = set()
    for path in files.values():
        with Image.open(path) as raw:
            shapes.add(raw.convert("RGB").size)
    assert len(shapes) > 1, (
        "naive load produced identical results for all orientations -- the "
        "fixture is not exercising the bug"
    )


def test_orientation_tag_stripped(pages, tmp_path):
    """Output carries no orientation tag, so no viewer re-rotates it."""
    _, files = pages
    loaded = pt_imageio.load(files[6])
    out = pt_imageio.save(loaded.pil, tmp_path / "out.jpg")
    with Image.open(out) as saved:
        exif = saved.getexif()
        assert exif.get(ORIENTATION_TAG, 1) == 1, "output still declares an orientation"


def test_pipeline_output_identical_across_orientations(pages, tmp_path):
    """The whole pipeline, end to end, is orientation-invariant.

    Identity translation isolates the geometry: any difference between the
    four outputs is placement, not wording.
    """
    _, files = pages
    outputs = {}
    for orientation, path in files.items():
        cfg = Config()
        cfg.output_dir = tmp_path / f"out{orientation}"
        cfg.source_lang, cfg.target_lang = "de", "de"
        cfg.overwrite = True
        result = Pipeline(cfg, translator=IdentityTranslator()).process_image(path)
        assert result.ok, result.error
        assert result.boxes > 0, f"no text detected in orientation {orientation}"
        with Image.open(result.output) as img:
            outputs[orientation] = np.asarray(img.convert("RGB"), dtype=np.int16)

    reference = outputs[1]
    for orientation, arr in outputs.items():
        assert arr.shape == reference.shape, (
            f"orientation {orientation} produced {arr.shape}, expected {reference.shape}"
        )
        diff = float(np.abs(arr - reference).mean())
        assert diff < 8.0, (
            f"orientation {orientation} rendered text in a different place "
            f"(mean diff {diff:.1f})"
        )


# ---------------------------------------------------------------------------
# The three runner-up hypotheses
# ---------------------------------------------------------------------------


def test_rotation_sign():
    """A box whose text descends to the right must rotate clockwise.

    Pinning this catches the y-down/y-up sign flip, whose signature is text
    leaning wrong by exactly -2*theta -- subtle enough at small angles to be
    mistaken for sloppy rendering rather than a bug.
    """
    theta = math.radians(12.0)
    dx, dy = 200 * math.cos(theta), 200 * math.sin(theta)
    box = TextBox([(100, 100), (100 + dx, 100 + dy),
                   (100 + dx - 20, 140 + dy), (80, 140)], "x")

    assert box.angle_deg == pytest.approx(12.0, abs=1.5)
    # Pillow rotates counter-clockwise for positive input, so matching a
    # clockwise-leaning box requires a negative value.
    assert degrees_for_pil(box.angle_rad) < 0
    assert degrees_for_pil(box.angle_rad) == pytest.approx(-12.0, abs=1.5)


def test_rotation_origin_is_box_centre():
    """Centroid, not image origin -- otherwise placement gains a translation."""
    box = TextBox([(500, 300), (700, 300), (700, 360), (500, 360)], "x")
    assert box.centre == (600.0, 330.0)


def test_quad_not_collapsed_to_rectangle():
    """A tilted quad keeps its angle and its true width and height.

    Treating the four points as an axis-aligned rectangle -- the hypothesis-4
    failure -- would report width 210 here instead of ~204, and angle 0.
    """
    theta = math.radians(20.0)
    w, h = 200.0, 50.0
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -math.sin(theta), math.cos(theta)
    p0 = (100.0, 100.0)
    quad = [
        p0,
        (p0[0] + w * ux, p0[1] + w * uy),
        (p0[0] + w * ux + h * vx, p0[1] + w * uy + h * vy),
        (p0[0] + h * vx, p0[1] + h * vy),
    ]
    box = TextBox(quad, "x")
    assert box.angle_deg == pytest.approx(20.0, abs=0.5)
    assert box.width == pytest.approx(w, abs=1.0)
    assert box.height == pytest.approx(h, abs=1.0)

    bx0, by0, bx1, by1 = box.bounds
    assert (bx1 - bx0) > box.width, "axis-aligned bounds should exceed the true width"


# ---------------------------------------------------------------------------
# Structural guard -- stops the bug growing back
# ---------------------------------------------------------------------------


def test_no_second_load_path():
    """Only imageio.py may decode an image.

    This is the test that keeps the fix permanent. The original bug was two
    code paths disagreeing about what "the image" is; the fix deleted the
    second one, and this fails if someone adds it back.
    """
    package = Path(__file__).resolve().parent.parent / "pt"
    offenders = []
    pattern = re.compile(r"\b(Image\.open|cv2\.imread|imageio\.imread|plt\.imread)\b")
    for source in package.glob("*.py"):
        if source.name == "imageio.py":
            continue
        for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                offenders.append(f"{source.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "image decoding outside pt/imageio.py -- this is exactly how the "
        "rotation bug returns:\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
