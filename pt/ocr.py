"""Two-tier OCR behind one interface."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from . import orient
from .config import Config, data_dir
from .geometry import TextBox, sort_reading_order
from .imageio import LoadedImage

log = logging.getLogger("pt.ocr")


class OcrEngine(ABC):
    name = "base"

    @abstractmethod
    def detect(self, image: LoadedImage) -> list[TextBox]:
        """Return detected text boxes in upright image coordinates."""

    @property
    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Tier 1 -- RapidOCR. CPU, ships inside the exe.
# ---------------------------------------------------------------------------


class RapidOcrEngine(OcrEngine):
    """ONNXRuntime port of PP-OCR."""

    name = "rapidocr"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._engine = None

    def _load(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:  # pragma: no cover
            try:
                from rapidocr import RapidOCR  # newer package name
            except ImportError as exc:
                raise RuntimeError(
                    "RapidOCR is not installed. Run:\n"
                    "    pip install rapidocr-onnxruntime"
                ) from exc
        log.info("loading RapidOCR (first call takes a few seconds)")
        self._engine = RapidOCR()
        return self._engine

    @property
    def available(self) -> bool:
        try:
            import rapidocr_onnxruntime  # noqa: F401
            return True
        except ImportError:
            try:
                import rapidocr  # noqa: F401
                return True
            except ImportError:
                return False

    def detect(self, image: LoadedImage) -> list[TextBox]:
        engine = self._load()
        # pass pixels, not the path, so EXIF handling stays in one place
        result = engine(image.bgr)

        # RapidOCR's return shape has moved between versions: older builds
        # return ``(list, elapse)``, newer ones return a result object.
        raw = result[0] if isinstance(result, tuple) else getattr(result, "boxes", result)
        if raw is None:
            return []

        boxes: list[TextBox] = []
        for item in raw:
            try:
                quad, text, conf = item[0], item[1], float(item[2])
            except (TypeError, IndexError, ValueError):
                continue
            if not text or not str(text).strip():
                continue
            try:
                box = TextBox(
                    quad=[(float(p[0]), float(p[1])) for p in quad],
                    text=str(text).strip(),
                    confidence=conf,
                )
            except (ValueError, TypeError):
                continue
            if conf < self.cfg.min_confidence:
                continue
            if min(box.height, box.width) < self.cfg.min_box_height or box.is_degenerate():
                continue
            boxes.append(box)

        # The detector reports where text is, not which way up it reads. A
        # vertical caption comes back as a tall quad with angle 0, which would
        # be typeset into a 60px-wide column. Resolve each box individually.
        if self.cfg.detect_orientation:
            boxes = orient.resolve_all(engine, image.bgr, boxes)
        return sort_reading_order(boxes)


# ---------------------------------------------------------------------------
# Tier 2 -- Unlimited-OCR. CUDA only, opt-in.
# ---------------------------------------------------------------------------


def cuda_available() -> bool:
    """True when a usable NVIDIA device is present."""
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


class UnlimitedOcrEngine(OcrEngine):
    """Adapter for the CUDA vision-language OCR model."""

    name = "unlimited"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._model = None
        self._processor = None
        self.model_id = os.environ.get("PT_UNLIMITED_MODEL", "")

    @property
    def available(self) -> bool:
        if not cuda_available():
            return False
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False
        # Only claim availability if the weights are already local, unless the
        # user explicitly asked for this engine. Otherwise "auto" would trigger
        # a multi-GB download on someone who just double-clicked an exe.
        if self.cfg.ocr_engine == "unlimited":
            return True
        return self._weights_present()

    def _weights_present(self) -> bool:
        cache = data_dir() / "unlimited-ocr"
        if cache.is_dir() and any(cache.rglob("*.safetensors")):
            return True
        hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        hub = os.path.join(hf_home, "hub")
        if not self.model_id or not os.path.isdir(hub):
            return False
        slug = "models--" + self.model_id.replace("/", "--")
        return os.path.isdir(os.path.join(hub, slug))

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        cache = str(data_dir() / "unlimited-ocr")
        log.info("loading Unlimited-OCR onto GPU (this is the slow part)")
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, cache_dir=cache
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=cache,
        ).eval().cuda()

    def _boxes_from_output(self, output, image: LoadedImage) -> list[TextBox]:
        """Normalise whatever the model returned into TextBoxes."""
        if output is None:
            return []
        if isinstance(output, (list, tuple)) and output and isinstance(output[0], dict):
            boxes = []
            for item in output:
                quad = item.get("quad") or item.get("polygon") or item.get("box")
                text = (item.get("text") or "").strip()
                if not text or not quad:
                    continue
                try:
                    if len(quad) == 4 and all(isinstance(p, (list, tuple)) for p in quad):
                        pts = [(float(p[0]), float(p[1])) for p in quad]
                    elif len(quad) == 4:  # x0,y0,x1,y1
                        x0, y0, x1, y1 = (float(v) for v in quad)
                        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    else:
                        continue
                    boxes.append(TextBox(pts, text, float(item.get("score", 1.0))))
                except (ValueError, TypeError):
                    continue
            return sort_reading_order(boxes)

        text = output if isinstance(output, str) else str(output)
        text = text.strip()
        if not text:
            return []
        w, h = image.size
        return [TextBox([(0, 0), (w, 0), (w, h), (0, h)], text, 1.0, meta={"page_level": True})]

    def detect(self, image: LoadedImage) -> list[TextBox]:
        try:
            self._load()
            import torch
            with torch.no_grad():
                output = self._model.infer(image=image.pil, processor=self._processor)
            return self._boxes_from_output(output, image)
        except Exception as exc:  # never raise into the pipeline
            log.warning("Unlimited-OCR failed (%s); falling back to RapidOCR", exc)
            return RapidOcrEngine(self.cfg).detect(image)


# ---------------------------------------------------------------------------


def get_engine(cfg: Config | None = None) -> OcrEngine:
    """Pick an engine per config, and report which one is used."""
    cfg = cfg or Config()
    choice = (cfg.ocr_engine or "auto").lower()

    if choice in ("rapid", "rapidocr", "cpu"):
        return RapidOcrEngine(cfg)
    if choice in ("unlimited", "gpu", "unlimited-ocr"):
        engine = UnlimitedOcrEngine(cfg)
        if engine.available:
            return engine
        log.warning("PT_OCR=unlimited but no usable CUDA device or weights; using RapidOCR")
        return RapidOcrEngine(cfg)

    # auto
    gpu = UnlimitedOcrEngine(cfg)
    if gpu.available:
        log.info("GPU detected and Unlimited-OCR weights present -- using the better engine")
        return gpu
    if cuda_available():
        log.info(
            "NVIDIA GPU detected. Unlimited-OCR would be more accurate here; "
            "its weights are several GB and are not downloaded automatically. "
            "Set PT_OCR=unlimited to enable it."
        )
    return RapidOcrEngine(cfg)
