"""Erase the original text, typeset the translation back into the same quad.

This stage decides whether the output looks real or looks like a ransom note.
Three things have to go right:

1. **Erasure.** Inpaint the text region from surrounding pixels, or fill it
   with the sampled background colour. Drawing over the top without erasing
   leaves the source language showing through the gaps in the new glyphs.
2. **Placement.** Rotate the rendered text about the quad's centre, by the
   negated image-space angle. Both halves of that sentence are load-bearing;
   see :func:`geometry.degrees_for_pil`.
3. **Shaping.** Persian and Arabic need contextual glyph shaping and bidi
   reordering *before* they are drawn. Skipping it -- which most naive
   implementations do -- produces disconnected letters in reverse order. It is
   the single most recognisable sign that nobody tested the RTL path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import Config, is_rtl
from .geometry import TextBox, degrees_for_pil
from .imageio import RESAMPLE_BICUBIC, LoadedImage
from .textshape import draw_kwargs, has_raqm, load_font, pick_font, prepare

log = logging.getLogger("pt.render")

# ---------------------------------------------------------------------------
# Colour sampling
# ---------------------------------------------------------------------------


def sample_colours(rgb: np.ndarray, box: TextBox) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return (ink, paper) RGB estimated from inside the box.

    Uses luminance percentiles rather than k-means: text is a minority of dark
    (or light) pixels against a dominant background, so the 15th and 85th
    percentiles separate them reliably and cost nothing. Which end is ink is
    decided by which is further from the median -- that keeps light-on-dark
    text working, which a hardcoded "ink is darker" assumption breaks.
    """
    x0, y0, x1, y1 = box.bounds
    h, w = rgb.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0), (255, 255, 255)

    patch = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if patch.size == 0:
        return (0, 0, 0), (255, 255, 255)

    lum = patch @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lo_cut, hi_cut = np.percentile(lum, [15, 85])
    dark = patch[lum <= lo_cut]
    light = patch[lum >= hi_cut]
    if dark.size == 0 or light.size == 0:
        mean = patch.mean(axis=0)
        return tuple(int(v) for v in mean), (255, 255, 255)

    dark_rgb = tuple(int(v) for v in np.median(dark, axis=0))
    light_rgb = tuple(int(v) for v in np.median(light, axis=0))

    # The background is whichever cluster holds more pixels near the median.
    median_lum = float(np.median(lum))
    if abs(median_lum - lum[lum <= lo_cut].mean()) < abs(median_lum - lum[lum >= hi_cut].mean()):
        return light_rgb, dark_rgb  # light-on-dark: ink is the light cluster
    return dark_rgb, light_rgb


# ---------------------------------------------------------------------------
# Erasure
# ---------------------------------------------------------------------------


def build_mask(shape: tuple[int, int], boxes: list[TextBox], pad: float = 2.0) -> np.ndarray:
    """White-on-black mask covering every text quad, slightly grown.

    The padding matters: detectors crop tight to the glyphs, and antialiased
    edge pixels just outside the quad survive erasure and read as a grey halo
    around where the old text was.
    """
    import cv2

    mask = np.zeros(shape[:2], dtype=np.uint8)
    for box in boxes:
        quad = box.expanded(pad).array.astype(np.int32)
        cv2.fillPoly(mask, [quad], 255)
    return mask


def erase(image_rgb: np.ndarray, boxes: list[TextBox], cfg: Config) -> np.ndarray:
    """Remove the source text. Returns a new array; the input is not modified."""
    mode = (cfg.erase_mode or "inpaint").lower()
    if mode == "none" or not boxes:
        return image_rgb.copy()

    if mode == "fill":
        out = image_rgb.copy()
        for box in boxes:
            _, paper = sample_colours(image_rgb, box)
            import cv2
            cv2.fillPoly(out, [box.expanded(2.0).array.astype(np.int32)], paper)
        return out

    import cv2
    mask = build_mask(image_rgb.shape, boxes, pad=2.0)
    bgr = image_rgb[:, :, ::-1].copy()
    # Telea is faster than Navier-Stokes and visually better on text, which is
    # thin structure over near-uniform paper -- the case it was designed for.
    healed = cv2.inpaint(bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return healed[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# Typesetting
# ---------------------------------------------------------------------------


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: float,
          kw: dict | None = None) -> list[str]:
    """Greedy word wrap against measured widths, not character counts.

    Measured, because a Persian word and an English word of the same character
    count differ in width by a factor of two, and because ``kw`` may carry
    ``direction="rtl"`` -- shaped text is not the same width as unshaped.
    """
    kw = kw or {}
    words = text.split()
    if not words:
        return []
    lines, current = [], words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font, **kw) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def fit_text(
    text: str,
    box_w: float,
    box_h: float,
    font_path: str,
    min_scale: float = 0.45,
    kw: dict | None = None,
) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """Largest font size at which ``text`` fits the box, with wrapping.

    Translations run longer than their source -- German to English is roughly
    +10%, English to Persian more -- so the box is almost always too small at
    the original size. Shrinking to ``min_scale`` of the box height and then
    accepting a slight overflow keeps the result readable; refusing to shrink
    would clip words, and shrinking without limit would produce text nobody
    can read.
    """
    kw = kw or {}
    probe = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(probe)

    start = max(8, int(box_h))
    floor = max(6, int(box_h * min_scale))
    best = None

    for size in range(start, floor - 1, -1):
        try:
            font = load_font(font_path, size)
        except OSError:
            break
        lines = _wrap(draw, text, font, box_w, kw)
        if not lines:
            continue
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        total_h = line_h * len(lines)
        widest = max(draw.textlength(line, font=font, **kw) for line in lines)
        if total_h <= box_h and widest <= box_w:
            return font, lines, line_h
        if best is None:
            best = (font, lines, line_h)

    # Nothing fit; use the smallest we tried and let it overflow slightly.
    font = load_font(font_path, floor)
    lines = _wrap(draw, text, font, box_w, kw)
    ascent, descent = font.getmetrics()
    return font, lines, ascent + descent


def draw_box(
    canvas: Image.Image,
    box: TextBox,
    text: str,
    font_path: str,
    ink: tuple[int, int, int],
    cfg: Config,
) -> None:
    """Render ``text`` into ``box`` on ``canvas``, respecting the box angle.

    The tile is rendered horizontally at the box's un-rotated dimensions, then
    rotated and pasted at the centroid. Rendering horizontally first is what
    lets the text wrap and centre correctly -- doing it in rotated space means
    every measurement has to account for the angle, and that is where sign
    errors breed.
    """
    if not text.strip():
        return

    # Dimensions in the box's OWN reading frame. For a vertical run these are
    # the quad's height and width swapped -- using the raw quad dimensions is
    # what crushed rotated captions into a sliver.
    box_w, box_h = box.text_size
    box_w = max(4.0, box_w)
    box_h = max(4.0, box_h)

    kw = draw_kwargs(cfg.target_lang)
    font, lines, line_h = fit_text(text, box_w, box_h, font_path, cfg.min_font_scale, kw)
    if not lines:
        return

    total_h = line_h * len(lines)
    # Pad the tile so rotation and antialiasing do not clip the glyphs.
    pad = int(max(4, line_h * 0.35))
    tile_w = int(box_w) + pad * 2
    tile_h = int(max(total_h, box_h)) + pad * 2

    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(tile)

    y = pad + max(0, (tile_h - pad * 2 - total_h) / 2)
    rtl = is_rtl(cfg.target_lang)
    for line in lines:
        line_w = tdraw.textlength(line, font=font, **kw)
        # RTL lines are right-aligned within the box, as the script reads.
        x = (tile_w - pad - line_w) if rtl else pad
        tdraw.text((x, y), line, font=font, fill=(*ink, 255), **kw)
        y += line_h

    # The one place an angle becomes a rotation call. render_angle_deg composes
    # the quad's page skew with the box's reading orientation; the sign lives in
    # degrees_for_pil so it is defined once and tested once.
    angle = box.render_angle_deg
    if abs(angle) > 0.1:
        tile = tile.rotate(angle, resample=RESAMPLE_BICUBIC, expand=True)

    cx, cy = box.centre
    left = int(round(cx - tile.width / 2))
    top = int(round(cy - tile.height / 2))
    canvas.alpha_composite(tile, dest=(max(0, left), max(0, top))) \
        if canvas.mode == "RGBA" else canvas.paste(tile, (left, top), tile)


def render(image: LoadedImage, boxes: list[TextBox], cfg: Config) -> Image.Image:
    """Full render: sample colours, erase, typeset every translated box."""
    # Font choice depends on the target script *and* the layout engine, so it
    # cannot be decided at import time. pick_font raises rather than returning
    # a font that would render the page as squares.
    font_path = pick_font(cfg.target_lang, cfg.font_path)
    if is_rtl(cfg.target_lang) and not has_raqm():
        # Not a problem, and worth saying so plainly: this path is verified to
        # produce identical output to the Raqm one. On Windows, Pillow ships a
        # Raqm that needs fribidi.dll present to activate, so this branch is the
        # normal case there rather than a degraded one.
        log.debug("Pillow has no Raqm; using the arabic-reshaper path for RTL layout")

    rgb = image.rgb
    usable = [b for b in boxes if (b.translated or "").strip()]

    # Colours must be sampled from the ORIGINAL pixels, before erasure --
    # afterwards the ink is gone and every box would sample paper on paper.
    inks = {id(b): sample_colours(rgb, b)[0] for b in usable}

    cleaned = erase(rgb, usable, cfg)
    canvas = Image.fromarray(cleaned).convert("RGB")

    if cfg.draw_debug_boxes:
        overlay = ImageDraw.Draw(canvas)
        for box in usable:
            overlay.polygon([tuple(p) for p in box.quad], outline=(255, 0, 0))

    for box in usable:
        draw_box(canvas, box, prepare(box.translated, cfg.target_lang),
                 font_path, inks[id(box)], cfg)

    return canvas
