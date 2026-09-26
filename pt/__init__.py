"""Photo Translator -- translate the text inside photos, offline."""

__version__ = "3.0.0"

from .config import Config
from .geometry import TextBox
from .pipeline import PageResult, Pipeline

__all__ = ["Config", "Pipeline", "PageResult", "TextBox", "__version__"]
