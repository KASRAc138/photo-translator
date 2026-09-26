"""Per-box text orientation."""

from __future__ import annotations

import logging

import numpy as np

from .geometry import TextBox

log = logging.getLogger("pt.orient")

# A crop taller than this many times its width is treated as a vertical run and
# probed at 90/270. Set from the observation that a single line of Latin text is
# rarely taller than it is wide -- even one character is roughly square.
VERTICAL_ASPECT = 1.35

# Below this, a horizontal box is re-probed at 180 in case the page is upside
# down. Above it, the detector's reading is trusted and no probe is run.
RECHECK_CONFIDENCE = 0.55


def warp_quad(image_bgr: np.ndarray, box: TextBox, pad: float = 2.0) -> np.ndarray | None:
    """Perspective-warp ``box``'s quad to an upright rectangle."""
    import cv2

    quad = box.expanded(pad)
    width = max(8, int(round(quad.width)))
    height = max(8, int(round(quad.height)))

    source = quad.array.astype(np.float32)
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    try:
        matrix = cv2.getPerspectiveTransform(source, target)
        return cv2.warpPerspective(
            image_bgr, matrix, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error as exc:
        log.debug("warp failed for box %r: %s", box.text[:20], exc)
        return None


def _rotate(crop: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a crop counter-clockwise by a multiple of 90."""
    import cv2

    if degrees % 360 == 0:
        return crop
    if degrees % 360 == 90:
        return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degrees % 360 == 180:
        return cv2.rotate(crop, cv2.ROTATE_180)
    return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)


def candidates_for(crop: np.ndarray, confidence: float) -> list[int]:
    """Which reading rotations are worth paying for on this crop."""
    height, width = crop.shape[:2]
    if height >= width * VERTICAL_ASPECT:
        # Tall and narrow: a vertical run. Which way it reads is genuinely
        # ambiguous -- a book spine and a rotated axis label lean opposite ways
        # -- so both are probed.
        return [90, 270, 0]
    if confidence < RECHECK_CONFIDENCE:
        return [0, 180]
    return [0]


def recognise(engine, crop: np.ndarray) -> tuple[str, float]:
    """Recognition only, no detection. Returns (text, confidence)."""
    try:
        result = engine(crop, use_det=False, use_rec=True, use_cls=False)
    except Exception as exc:
        log.debug("recognition pass failed: %s", exc)
        return "", 0.0

    raw = result[0] if isinstance(result, tuple) else result
    if not raw:
        return "", 0.0
    try:
        first = raw[0]
        return str(first[0] or "").strip(), float(first[1])
    except (TypeError, IndexError, ValueError):
        return "", 0.0


def resolve(engine, image_bgr: np.ndarray, box: TextBox) -> TextBox:
    """Decide ``box``'s reading rotation and refine its text."""
    crop = warp_quad(image_bgr, box)
    if crop is None or crop.size == 0:
        box.orientation = 0
        return box

    options = candidates_for(crop, box.confidence)
    if options == [0]:
        box.orientation = 0
        return box

    # Score every candidate with the SAME recogniser on the SAME crop, and
    # compare only those scores against each other.
    #
    # don't use the detector's confidence as baseline: RapidOCR rotates
    # internally, so it reports ~0.99 even for sideways boxes
    scores: list[tuple[float, int, str]] = []
    for rotation in options:
        text, conf = recognise(engine, _rotate(crop, rotation))
        if text:
            scores.append((conf, rotation, text))

    if not scores:
        # Nothing read at any angle. Keep the detector's answer rather than
        # discarding a box it was confident about.
        box.orientation = 0
        return box

    best_conf, best_rotation, best_text = max(scores, key=lambda s: s[0])

    # Prefer upright on a near-tie: a genuinely rotated run beats its upright
    # reading by a wide margin (typically 1.00 against 0.00), so a close call
    # means the probe is guessing.
    upright = next((s for s in scores if s[1] == 0), None)
    if upright and best_rotation != 0 and best_conf - upright[0] < 0.15:
        best_conf, best_rotation, best_text = upright

    box.orientation = best_rotation % 360
    box.confidence = best_conf
    # Only take the probe's text when it actually read something better; the
    # detector's own string is usually right even when its geometry is not.
    if best_conf >= box.confidence or not box.text.strip():
        box.text = best_text
    if box.orientation:
        log.debug("box %r reads at %d degrees (conf %.2f)",
                  box.text[:24], box.orientation, best_conf)
        box.meta["reoriented"] = True
    return box


def resolve_all(engine, image_bgr: np.ndarray, boxes: list[TextBox]) -> list[TextBox]:
    """Run :func:`resolve` over every box, dropping any left unreadable."""
    resolved = [resolve(engine, image_bgr, box) for box in boxes]
    kept = [b for b in resolved if b.text.strip()]
    rotated = sum(1 for b in kept if b.orientation)
    if rotated:
        log.info("%d of %d text runs are rotated", rotated, len(kept))
    return kept
