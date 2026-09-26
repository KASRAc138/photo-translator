"""Translation, pluggable, offline by default."""

from __future__ import annotations

import logging
import re
import sys
import types
from abc import ABC, abstractmethod

from .config import Config

log = logging.getLogger("pt.translate")

# Argos's hub language. Every pack goes to or from English; two-hop routes
# through it are the norm, not the exception.
PIVOT = "en"


# Sentence terminators across the scripts this app sees. Persian and Urdu use
# U+061F and U+06D4, Devanagari U+0964; a splitter that only knows "." silently
# treats a whole Persian paragraph as one sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?\u061f\u06d4\u0964\u3002])\s+|\n{2,}")


def _split_sentences(text: str) -> list[str]:
    """Sentence split good enough for OCR output, with no model and no download."""
    parts = [p.strip() for p in _SENTENCE_END.split(text or "") if p and p.strip()]
    return parts or ([text] if text else [])


class _StubSentence:
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


class _StubDoc:
    __slots__ = ("sentences", "text")

    def __init__(self, text: str):
        self.text = text
        self.sentences = [_StubSentence(s) for s in _split_sentences(text)]


class _StubPipeline:
    """Stands in for ``stanza.Pipeline`` with the surface Argos actually uses."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __call__(self, text: str) -> _StubDoc:
        return _StubDoc(text)


def _install_sbd_stubs() -> None:
    """Let argostranslate import and run without stanza, so the exe stays small."""
    import os
    if os.environ.get("PT_USE_STANZA", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    if getattr(sys.modules.get("stanza"), "__version__", "") == "0.0.0-stub":
        return

    stub = types.ModuleType("stanza")
    stub.__version__ = "0.0.0-stub"
    stub.__doc__ = (
        "Minimal functional stand-in installed by Photo Translator. Provides "
        "regex sentence splitting so Argos can run without torch."
    )
    stub.Pipeline = _StubPipeline
    stub.download = lambda *a, **k: None
    sys.modules["stanza"] = stub
    log.debug("stanza absent; installed a functional regex-based stand-in")


# skip part numbers, dates, labels, measurements (e.g. "KW07")
_NOT_WORDS = re.compile(r"^[\W\d_]*$|^[A-Z]{1,4}[\d.\-/]+[A-Za-z]*$")


def is_translatable(text: str) -> bool:
    """False for tokens that carry no language -- numbers, codes, punctuation."""
    stripped = (text or "").strip()
    if len(stripped) < 2:
        return False
    if _NOT_WORDS.match(stripped):
        return False
    # Needs at least two consecutive letters somewhere to be a word.
    return bool(re.search(r"[^\W\d_]{2,}", stripped))


class Translator(ABC):
    name = "base"

    @abstractmethod
    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        """Translate a batch."""

    def close(self) -> None:
        pass


class ArgosTranslator(Translator):
    name = "argos"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._ready: set[tuple[str, str]] = set()

    # -- package management ----------------------------------------------

    @staticmethod
    def installed_pairs() -> set[tuple[str, str]]:
        _install_sbd_stubs()
        import argostranslate.package as package
        return {(p.from_code, p.to_code) for p in package.get_installed_packages()}

    def ensure_packages(self, source: str, target: str) -> bool:
        """Install what is needed for ``source -> target``, pivoting if required."""
        if source == target:
            return True
        if (source, target) in self._ready:
            return True

        _install_sbd_stubs()
        import argostranslate.package as package

        have = self.installed_pairs()
        if (source, target) in have:
            self._ready.add((source, target))
            return True
        if (source, PIVOT) in have and (PIVOT, target) in have:
            self._ready.add((source, target))
            return True

        if self.cfg.offline:
            log.error(
                "No Argos pack for %s->%s and PT_OFFLINE is set. "
                "Install one on a connected machine first.", source, target
            )
            return False

        try:
            package.update_package_index()
            available = package.get_available_packages()
        except Exception as exc:
            log.error("Could not reach the Argos package index: %s", exc)
            return False

        index = {(p.from_code, p.to_code): p for p in available}

        # Prefer a direct pack; fall back to the two-hop route through English.
        if (source, target) in index:
            needed = [(source, target)]
        elif (source, PIVOT) in index and (PIVOT, target) in index:
            log.info("No direct %s->%s pack; pivoting through %s", source, target, PIVOT)
            needed = [(source, PIVOT), (PIVOT, target)]
        else:
            log.error("Argos has no route from %s to %s", source, target)
            return False

        for pair in needed:
            if pair in have:
                continue
            pkg = index[pair]
            log.info("downloading language pack %s->%s (~100 MB, once)", *pair)
            try:
                package.install_from_path(pkg.download())
            except Exception as exc:
                log.error("Language pack %s->%s failed to download: %s", *pair, exc)
                return False

        self._ready.add((source, target))
        return True

    # -- routing ----------------------------------------------------------

    @staticmethod
    def chain_for(source: str, target: str) -> list | None:
        """Translation objects to apply in order, or None if there is no route."""
        _install_sbd_stubs()
        import argostranslate.translate as translate_mod

        try:
            languages = {lang.code: lang for lang in translate_mod.get_installed_languages()}
        except Exception as exc:
            log.error("could not read installed Argos languages: %s", exc)
            return None

        src, dst = languages.get(source), languages.get(target)
        if src is None or dst is None:
            return None

        try:
            direct = src.get_translation(dst)
        except Exception:
            direct = None
        if direct is not None:
            return [direct]

        pivot = languages.get(PIVOT)
        if pivot is None or pivot is src or pivot is dst:
            return None
        try:
            first, second = src.get_translation(pivot), pivot.get_translation(dst)
        except Exception:
            return None
        if first is not None and second is not None:
            log.info("routing %s->%s through %s", source, target, PIVOT)
            return [first, second]
        return None

    # -- detection --------------------------------------------------------

    @staticmethod
    def detect_language(text: str, fallback: str = "en") -> str:
        """Best-effort source language guess."""
        sample = (text or "").strip()
        if len(sample) < 8:
            return fallback
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            return detect(sample)
        except Exception:
            return fallback

    # -- the work ---------------------------------------------------------

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        if not texts:
            return []
        if source == target:
            return list(texts)
        if not self.ensure_packages(source, target):
            return list(texts)  # pass the original through rather than blanking the image

        chain = self.chain_for(source, target)
        if chain is None:
            # Reported once for the whole page, not once per line. The old
            # behaviour raised an AttributeError per string, which buried the
            # actual problem under dozens of identical tracebacks.
            log.error(
                "No Argos route from %s to %s even though the packs report as "
                "installed. Try:  python PhotoTranslator.py --setup --from %s --to %s",
                source, target, source, target,
            )
            return list(texts)

        out: list[str] = []
        for text in texts:
            stripped = (text or "").strip()
            if not stripped or not is_translatable(stripped):
                out.append(text)      # codes, numbers and labels pass through unchanged
                continue
            try:
                value = stripped
                for hop in chain:
                    value = hop.translate(value)
                out.append(value if value and value.strip() else text)
            except Exception as exc:
                log.warning("translation failed for %r: %s", stripped[:40], exc)
                out.append(text)
        return out


class IdentityTranslator(Translator):
    """Passthrough. For testing the OCR and render stages in isolation."""

    name = "none"

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        return list(texts)


def get_translator(cfg: Config | None = None) -> Translator:
    cfg = cfg or Config()
    choice = (cfg.translate_engine or "argos").lower()
    if choice in ("none", "off", "identity"):
        return IdentityTranslator()
    return ArgosTranslator(cfg)
