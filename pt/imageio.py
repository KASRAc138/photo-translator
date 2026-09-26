"""The single image load path."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Pillow >= 9.1 moved the resampling enums.
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover - old Pillow
    RESAMPLE_BICUBIC = Image.BICUBIC

# Scanned book pages are legitimately huge. Raise the decompression-bomb
# ceiling rather than letting Pillow refuse a 12000px archival scan, but keep
# it finite so a malicious file cannot exhaust memory.
Image.MAX_IMAGE_PIXELS = 512_000_000

SUPPORTED_SUFFIXES = {
    ".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".tif", ".tiff", ".webp",
}

# EXIF tag 0x0112 == Orientation.
_ORIENTATION_TAG = 0x0112


@dataclass
class LoadedImage:
    """One decoded image, already upright, with its provenance attached."""

    path: Path
    pil: Image.Image           # RGB, upright, EXIF tag stripped
    original_orientation: int  # what the file claimed, for logging (1 == none)

    @property
    def size(self) -> tuple[int, int]:
        return self.pil.size

    @property
    def width(self) -> int:
        return self.pil.size[0]

    @property
    def height(self) -> int:
        return self.pil.size[1]

    @property
    def bgr(self) -> np.ndarray:
        """OpenCV-order ndarray of the *same* upright pixels."""
        return np.asarray(self.pil, dtype=np.uint8)[:, :, ::-1].copy()

    @property
    def rgb(self) -> np.ndarray:
        return np.asarray(self.pil, dtype=np.uint8).copy()

    def was_rotated(self) -> bool:
        return self.original_orientation not in (0, 1)


def _read_orientation(img: Image.Image) -> int:
    """Return the declared EXIF orientation, or 1 when absent/unreadable."""
    try:
        exif = img.getexif()
    except Exception:
        return 1
    if not exif:
        return 1
    try:
        value = exif.get(_ORIENTATION_TAG, 1)
    except Exception:
        return 1
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 1
    return value if 1 <= value <= 8 else 1


def load(path: str | Path) -> LoadedImage:
    """Decode ``path`` upright."""
    path = Path(path)
    with Image.open(path) as raw:
        raw.load()
        declared = _read_orientation(raw)
        upright = ImageOps.exif_transpose(raw)
        if upright is None:  # exif_transpose returns None on some Pillow versions
            upright = raw.copy()
        upright = upright.convert("RGB")

    # Strip EXIF so nothing downstream -- including the image viewer the user
    # opens the output in -- applies the rotation a second time. This is the
    # other half of the bug: correcting the pixels but keeping the tag means a
    # viewer that honours EXIF re-rotates the already-corrected image.
    upright.info.pop("exif", None)
    if "exif" in getattr(upright, "info", {}):  # defensive
        del upright.info["exif"]

    return LoadedImage(path=path, pil=upright, original_orientation=declared)


def load_from_bytes(data: bytes, name: str = "<memory>") -> LoadedImage:
    """Same contract as :func:`load`, for data that never hit disk."""
    with Image.open(io.BytesIO(data)) as raw:
        raw.load()
        declared = _read_orientation(raw)
        upright = ImageOps.exif_transpose(raw) or raw.copy()
        upright = upright.convert("RGB")
    upright.info.pop("exif", None)
    return LoadedImage(path=Path(name), pil=upright, original_orientation=declared)


def save(img: Image.Image, path: str | Path, quality: int = 95) -> Path:
    """Write an image out with **no** orientation metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB") if img.mode not in ("RGB", "L") else img
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img.save(path, "JPEG", quality=quality, subsampling=0, optimize=True)
    elif suffix == ".png":
        img.save(path, "PNG", optimize=True)
    else:
        img.save(path)
    return path


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def iter_images(folder: str | Path):
    """Yield supported image paths in ``folder``, sorted, non-recursive."""
    folder = Path(folder)
    if not folder.is_dir():
        return
    for entry in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_file() and is_supported(entry):
            yield entry
