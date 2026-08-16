"""Command line / double-click entry point.

Designed so that running the .exe with no arguments does the obvious thing:
create ``input/`` and ``output/`` next to itself, say where they are, translate
whatever is in ``input/``, and wait for a keypress so the window does not
vanish before the user can read it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config, app_dir
from .pipeline import Pipeline

BANNER = r"""
  Photo Translator
  ----------------
"""


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="PhotoTranslator",
        description="Translate the text inside photos, in place, offline.",
    )
    p.add_argument("images", nargs="*", type=Path,
                   help="specific files to translate (default: everything in input/)")
    p.add_argument("--from", dest="source", metavar="LANG",
                   help="source language code, or 'auto' (default: auto)")
    p.add_argument("--to", dest="target", metavar="LANG",
                   help="target language code (default: en)")
    p.add_argument("--input", type=Path, help="input folder")
    p.add_argument("--output", type=Path, help="output folder")
    p.add_argument("--ocr", choices=["auto", "rapidocr", "unlimited"],
                   help="OCR tier (default: auto)")
    p.add_argument("--erase", choices=["inpaint", "fill", "none"],
                   help="how the original text is removed (default: inpaint)")
    p.add_argument("--font", type=Path, help="TrueType font to typeset with")
    p.add_argument("--debug-boxes", action="store_true",
                   help="outline detected quads in red, to check placement")
    p.add_argument("--overwrite", action="store_true",
                   help="replace existing outputs instead of numbering them")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-pause", action="store_true",
                   help="do not wait for a keypress when finished")
    p.add_argument("--setup", action="store_true",
                   help="create folders, download the language pack, and exit")
    p.add_argument("--doctor", action="store_true",
                   help="report what is installed and working, then exit")
    p.add_argument("--batch", action="store_true",
                   help="translate input/ from the command line instead of opening the UI")
    p.add_argument("--port", type=int, default=8731, help="UI port (default: 8731)")
    p.add_argument("--no-browser", action="store_true",
                   help="start the UI but do not open a browser")
    return p


def apply_args(cfg: Config, args: argparse.Namespace) -> Config:
    if args.source:
        cfg.source_lang = args.source
    if args.target:
        cfg.target_lang = args.target
    if args.input:
        cfg.input_dir = args.input
    if args.output:
        cfg.output_dir = args.output
    if args.ocr:
        cfg.ocr_engine = args.ocr
    if args.erase:
        cfg.erase_mode = args.erase
    if args.font:
        cfg.font_path = str(args.font)
    if args.debug_boxes:
        cfg.draw_debug_boxes = True
    if args.overwrite:
        cfg.overwrite = True
    if args.quiet:
        cfg.verbose = False
    return cfg


def doctor(cfg: Config) -> int:
    """Report the state of everything that can silently degrade.

    Each of these has a failure mode that produces wrong output rather than an
    error: no Raqm reshapes differently, a font without Arabic glyphs draws
    empty boxes, a missing language pack passes the source text straight
    through. Worth being able to check in one command.
    """
    ok = True
    print("  Photo Translator -- self check\n")

    import platform
    frozen = getattr(sys, "frozen", False)
    print(f"  python      {platform.python_version()} on {platform.system()} "
          f"({'frozen exe' if frozen else 'source'})")

    # -- imaging ------------------------------------------------------
    try:
        from PIL import Image, features
        raqm = bool(features.check("raqm"))
        print(f"  pillow      {Image.__version__}   raqm/harfbuzz: "
              + ("yes" if raqm else "no -- using the arabic-reshaper path (verified equivalent)"))
    except Exception as exc:
        print(f"  pillow      MISSING ({exc})"); ok = False

    # -- fonts --------------------------------------------------------
    from .textshape import coverage, pick_font
    for lang, label in (("en", "latin"), (cfg.target_lang, cfg.target_lang)):
        try:
            path = pick_font(lang)
            score = coverage(path, lang)
            flag = "ok" if score > 0.95 else f"only {score:.0%} coverage"
            print(f"  font {label:<7}{Path(path).name}  ({flag})")
            if score <= 0.95:
                ok = False
        except Exception as exc:
            print(f"  font {label:<7}NONE USABLE -- {exc}"); ok = False

    # -- OCR ----------------------------------------------------------
    from .ocr import RapidOcrEngine, cuda_available
    rapid = RapidOcrEngine(cfg)
    print(f"  ocr cpu     rapidocr: {'available' if rapid.available else 'MISSING'}")
    if not rapid.available:
        ok = False
    print(f"  ocr gpu     cuda: {'present' if cuda_available() else 'not present (CPU tier is used)'}")

    # -- translation --------------------------------------------------
    from .translate import ArgosTranslator
    try:
        pairs = sorted(ArgosTranslator.installed_pairs())
        print(f"  argos packs {', '.join(f'{a}->{b}' for a, b in pairs) if pairs else 'none installed yet'}")
        source = cfg.source_lang if cfg.source_lang != "auto" else "de"
        chain = ArgosTranslator.chain_for(source, cfg.target_lang)
        if chain is None:
            print(f"  route       {source} -> {cfg.target_lang}: NOT AVAILABLE")
            print(f"              fix with:  PhotoTranslator --setup --from {source} --to {cfg.target_lang}")
            ok = False
        else:
            hops = "direct" if len(chain) == 1 else f"via {len(chain)} hops through English"
            print(f"  route       {source} -> {cfg.target_lang}: {hops}")
    except Exception as exc:
        print(f"  argos       ERROR {exc}"); ok = False

    print(f"\n  {'All good.' if ok else 'Some checks failed -- see above.'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_args(Config(), args)
    setup_logging(cfg.verbose)

    print(BANNER)
    in_dir, out_dir = cfg.ensure_folders()
    print(cfg.describe())
    print()

    if args.doctor:
        return doctor(cfg)

    if args.setup:
        from .translate import ArgosTranslator
        print("Downloading the language pack (once) ...")
        source = cfg.source_lang if cfg.source_lang != "auto" else "en"
        ok = ArgosTranslator(cfg).ensure_packages(source, cfg.target_lang)
        print("Ready." if ok else "Could not install the language pack -- see the messages above.")
        return 0 if ok else 1

    # The UI is the default: someone who double-clicks the exe wants a window,
    # not a folder convention they have to be told about. --batch keeps the
    # scriptable path for automation and for the KB integration.
    if not args.batch and not args.images:
        from .webui import serve
        serve(cfg, port=args.port, open_browser=not args.no_browser)
        return 0

    pipeline = Pipeline(cfg)

    if args.images:
        results = [pipeline.process_image(p) for p in args.images]
    else:
        results = pipeline.process_folder()

    ok = sum(1 for r in results if r.ok)
    failed = len(results) - ok
    print()
    if not results:
        print(f"Nothing to do. Put photos in:\n    {in_dir}\nthen run this again.")
    else:
        print(f"Done. {ok} translated"
              + (f", {failed} failed" if failed else "")
              + f".\nOutput folder:\n    {out_dir}")

    if not args.no_pause and sys.stdin is not None and sys.stdin.isatty():
        try:
            input("\nPress Enter to close ...")
        except (EOFError, KeyboardInterrupt):
            pass
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
