"""Quad geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


@dataclass
class TextBox:
    """A detected piece of text and the quad it occupies."""

    quad: list[Point]
    text: str
    confidence: float = 1.0
    translated: str | None = None
    # Counter-clockwise rotation, in degrees, that makes this box's warped crop
    # read correctly. 0 for ordinary horizontal text; 90 or 270 for a vertical
    # run. Set by pt.orient -- the detector cannot be trusted to report it, and
    # a page routinely mixes several values. See pt/orient.py.
    orientation: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.quad) != 4:
            raise ValueError(f"quad needs exactly 4 points, got {len(self.quad)}")
        self.quad = [(float(x), float(y)) for x, y in self.quad]

    # -- derived geometry -------------------------------------------------

    @property
    def array(self) -> np.ndarray:
        return np.array(self.quad, dtype=np.float32)

    @property
    def centre(self) -> Point:
        """Centroid."""
        a = self.array
        return (float(a[:, 0].mean()), float(a[:, 1].mean()))

    @property
    def angle_rad(self) -> float:
        """Angle of the text baseline, in **image** coordinates (y grows down)."""
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.quad
        top = math.atan2(y1 - y0, x1 - x0)
        bottom = math.atan2(y2 - y3, x2 - x3)
        # Average as unit vectors so the wrap at +-pi cannot produce a bogus mean.
        sx = math.cos(top) + math.cos(bottom)
        sy = math.sin(top) + math.sin(bottom)
        if sx == 0 and sy == 0:
            return top
        return math.atan2(sy, sx)

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle_rad)

    @property
    def width(self) -> float:
        """Length along the reading direction, not the image x-axis."""
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.quad
        top = math.hypot(x1 - x0, y1 - y0)
        bottom = math.hypot(x2 - x3, y2 - y3)
        return (top + bottom) / 2.0

    @property
    def height(self) -> float:
        """Length across the reading direction (the cap-to-descender span)."""
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.quad
        left = math.hypot(x3 - x0, y3 - y0)
        right = math.hypot(x2 - x1, y2 - y1)
        return (left + right) / 2.0

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Axis-aligned integer bounding box (x0, y0, x1, y1). For masks only."""
        a = self.array
        return (
            int(math.floor(a[:, 0].min())),
            int(math.floor(a[:, 1].min())),
            int(math.ceil(a[:, 0].max())),
            int(math.ceil(a[:, 1].max())),
        )

    @property
    def area(self) -> float:
        """Shoelace area. Robust to point order winding."""
        a = self.array
        x, y = a[:, 0], a[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    @property
    def is_vertical(self) -> bool:
        return self.orientation in (90, 270)

    @property
    def text_size(self) -> tuple[float, float]:
        """(length, thickness) of the text run *along its own reading direction*."""
        if self.is_vertical:
            return self.height, self.width
        return self.width, self.height

    @property
    def render_angle_deg(self) -> float:
        """Total rotation for a horizontally-rendered tile of this box's text."""
        return degrees_for_pil(self.angle_rad) - self.orientation

    def is_degenerate(self, min_side: float = 3.0) -> bool:
        return self.width < min_side or self.height < min_side

    def expanded(self, pad: float) -> "TextBox":
        """Grow the quad outward from its centre by ``pad`` pixels each way."""
        cx, cy = self.centre
        out = []
        for x, y in self.quad:
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy) or 1.0
            out.append((x + dx / d * pad, y + dy / d * pad))
        return TextBox(out, self.text, self.confidence, self.translated,
                       self.orientation, dict(self.meta))


def degrees_for_pil(angle_rad: float) -> float:
    """Convert an image-space angle to the value Pillow's ``rotate`` wants."""
    return -math.degrees(angle_rad)


def sort_reading_order(boxes: list[TextBox], line_tol_ratio: float = 0.6) -> list[TextBox]:
    """Order boxes top-to-bottom, then left-to-right within a line."""
    if not boxes:
        return []
    typical = float(np.median([b.height for b in boxes])) or 1.0
    tol = typical * line_tol_ratio

    remaining = sorted(boxes, key=lambda b: b.centre[1])
    lines: list[list[TextBox]] = []
    for box in remaining:
        placed = False
        for line in lines:
            if abs(line[0].centre[1] - box.centre[1]) <= tol:
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])

    ordered: list[TextBox] = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda b: b.centre[0]))
    return ordered
