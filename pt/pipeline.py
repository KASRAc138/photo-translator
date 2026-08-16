"""The pipeline: folder in, translated folder out.

Order is load -> detect -> translate -> render -> save, and the loaded image
object is threaded through all of it. No stage re-opens the file.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from . import imageio
from .config import Config
from .geometry import TextBox
from .ocr import OcrEngine, get_engine
from .render import render
from .translate import ArgosTranslator, Translator, get_translator

log = logging.getLogger("pt.pipeline")


@dataclass
class PageResult:
    source: Path
    output: Path | None
    boxes: int
    seconds: float
    error: str | None = None
    rotated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


class Pipeline:
    def __init__(self, cfg: Config | None = None,
                 ocr: OcrEngine | None = None,
                 translator: Translator | None = None):
        self.cfg = cfg or Config()
        self.ocr = ocr or get_engine(self.cfg)
        self.translator = translator or get_translator(self.cfg)

    # -- one image --------------------------------------------------------

    def process_image(self, path: Path) -> PageResult:
        started = time.time()
        try:
            image = imageio.load(path)
        except Exception as exc:
            return PageResult(path, None, 0, time.time() - started, f"could not read: {exc}")

        if image.was_rotated():
            log.info("%s carried EXIF orientation %d; normalised once at load",
                     path.name, image.original_orientation)

        try:
            boxes: list[TextBox] = self.ocr.detect(image)
        except Exception as exc:
            return PageResult(path, None, 0, time.time() - started, f"OCR failed: {exc}",
                              image.was_rotated())

        if not boxes:
            log.info("%s: no text found", path.name)
            out = self._output_path(path)
            imageio.save(image.pil, out, self.cfg.jpeg_quality)
            return PageResult(path, out, 0, time.time() - started, None, image.was_rotated())

        source = self.cfg.source_lang
        if source == "auto":
            joined = " ".join(b.text for b in boxes[:12])
            source = ArgosTranslator.detect_language(joined, fallback="en")
            log.info("%s: detected source language %s", path.name, source)

        translations = self.translator.translate(
            [b.text for b in boxes], source, self.cfg.target_lang
        )
        for box, translated in zip(boxes, translations):
            box.translated = translated

        try:
            canvas = render(image, boxes, self.cfg)
        except Exception as exc:
            return PageResult(path, None, len(boxes), time.time() - started,
                              f"render failed: {exc}", image.was_rotated())

        out = self._output_path(path)
        imageio.save(canvas, out, self.cfg.jpeg_quality)
        return PageResult(path, out, len(boxes), time.time() - started, None, image.was_rotated())

    def _output_path(self, source: Path) -> Path:
        stem = source.stem
        suffix = ".png" if source.suffix.lower() == ".png" else ".jpg"
        out = self.cfg.output_dir / f"{stem}_{self.cfg.target_lang}{suffix}"
        if out.exists() and not self.cfg.overwrite:
            n = 2
            while (self.cfg.output_dir / f"{stem}_{self.cfg.target_lang}_{n}{suffix}").exists():
                n += 1
            out = self.cfg.output_dir / f"{stem}_{self.cfg.target_lang}_{n}{suffix}"
        return out

    # -- a folder ---------------------------------------------------------

    def process_folder(self) -> list[PageResult]:
        in_dir, out_dir = self.cfg.ensure_folders()
        images = list(imageio.iter_images(in_dir))
        if not images:
            log.warning("No images in %s -- drop some photos there and run again.", in_dir)
            return []

        log.info("Found %d image%s in %s", len(images), "" if len(images) == 1 else "s", in_dir)
        results = []
        for i, path in enumerate(images, 1):
            log.info("[%d/%d] %s", i, len(images), path.name)
            result = self.process_image(path)
            if result.ok:
                log.info("      -> %s  (%d boxes, %.1fs)",
                         result.output.name, result.boxes, result.seconds)
            else:
                log.error("      !! %s", result.error)
            results.append(result)
        return results
