"""Photo Translator -- translate the text inside photos, offline.

Public surface is deliberately small:

    from pt import Config, Pipeline
    Pipeline(Config(target_lang="fa")).process_folder()

Everything else is an implementation detail, with one exception worth knowing:
``pt.imageio.load`` is the *only* sanctioned way to read an image anywhere in
this package. See that module for why.
"""

__version__ = "3.0.0"

from .config import Config
from .geometry import TextBox
from .pipeline import PageResult, Pipeline

__all__ = ["Config", "Pipeline", "PageResult", "TextBox", "__version__"]
